# app.py

from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
import os
import uuid
import json
from celery import Celery, states
from celery.exceptions import Ignore
import whisper
from pydub import AudioSegment
import tempfile
import traceback
import torch
import warnings
from pathlib import Path
from typing import List, Dict, Any
from celery.signals import worker_init
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    # Base configuration
    UPLOAD_FOLDER = Path('uploads')
    CELERY_BROKER_URL = 'redis://localhost:6379/0'
    RESULT_BACKEND = 'redis://localhost:6379/0'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB limit
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}
    WHISPER_MODEL = "base"
    MAX_CHUNK_DURATION = 10  # seconds

    @staticmethod
    def init_folders():
        """Initialize necessary folders"""
        folders = [
            Config.UPLOAD_FOLDER,
            Config.UPLOAD_FOLDER / "temp"
        ]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

# Flask app configuration
app = Flask(__name__)
app.config.from_object(Config)

# Initialize Celery
celery = Celery(app.name)
celery.conf.update(
    broker_url=app.config['CELERY_BROKER_URL'],
    result_backend=app.config['RESULT_BACKEND'],
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    task_track_started=True
)

class VideoProcessor:
    def __init__(self, video_path: str, processing_id: str):
        self.video_path = video_path
        self.processing_id = processing_id
        self.output_dir = Config.UPLOAD_FOLDER / processing_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def initialize_model(self):
        """Initialize Whisper model"""
        try:
            self.model = whisper.load_model(Config.WHISPER_MODEL, device=self.device)
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    def process_video(self, task) -> List[Dict[str, Any]]:
        """Main video processing pipeline"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            task.update_state(state='PROCESSING', meta={'step': 'Extracting audio'})
            transcript = self._transcribe_audio()
            
            task.update_state(state='PROCESSING', meta={'step': 'Processing audio chunks'})
            chunks = self._process_chunks(transcript)
            
            self._save_metadata(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in process_video: {str(e)}")
            raise RuntimeError(f"Video processing failed: {str(e)}")

    def _transcribe_audio(self) -> Dict[str, Any]:
        """Transcribe audio from video"""
        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', dir=Config.UPLOAD_FOLDER / "temp", delete=False) as tmp_audio:
                temp_audio_path = tmp_audio.name
                audio = AudioSegment.from_file(self.video_path)
                audio.export(temp_audio_path, format="wav")
                self.initialize_model()
                result = self.model.transcribe(temp_audio_path)
                return result
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    def _process_chunks(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process and save audio chunks"""
        try:
            chunks = self._semantic_chunking(transcript["segments"])
            audio = AudioSegment.from_file(self.video_path)
            
            output = []
            for i, chunk in enumerate(chunks):
                chunk_data = self._save_chunk(i, chunk, audio)
                output.append(chunk_data)
            
            return output
        except Exception as e:
            logger.error(f"Error in process_chunks: {str(e)}")
            raise RuntimeError(f"Chunk processing failed: {str(e)}")

    @staticmethod
    def _semantic_chunking(segments: List[Dict[str, Any]], max_duration: int = None) -> List[List[Dict[str, Any]]]:
        """Chunk segments semantically"""
        if max_duration is None:
            max_duration = Config.MAX_CHUNK_DURATION
            
        chunks = []
        current_chunk = []
        current_duration = 0
        
        for segment in segments:
            seg_duration = segment['end'] - segment['start']
            if current_duration + seg_duration <= max_duration:
                current_chunk.append(segment)
                current_duration += seg_duration
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [segment]
                current_duration = seg_duration
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _save_chunk(self, index: int, chunk: List[Dict[str, Any]], audio: AudioSegment) -> Dict[str, Any]:
        """Save individual audio chunk and return metadata"""
        try:
            start_time = int(chunk[0]['start'] * 1000)
            end_time = int(chunk[-1]['end'] * 1000)
            chunk_audio = audio[start_time:end_time]
            
            chunk_filename = f"chunk_{index+1}.wav"
            chunk_path = self.output_dir / chunk_filename
            chunk_audio.export(str(chunk_path), format="wav")
            
            return {
                "chunk_id": index + 1,
                "start_time": chunk[0]['start'],
                "end_time": chunk[-1]['end'],
                "text": " ".join([s['text'].strip() for s in chunk]),
                "audio_file": chunk_filename
            }
        except Exception as e:
            logger.error(f"Error saving chunk {index}: {str(e)}")
            raise RuntimeError(f"Failed to save chunk {index}: {str(e)}")

    def _save_metadata(self, chunks: List[Dict[str, Any]]):
        """Save processing metadata"""
        try:
            metadata_path = self.output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(chunks, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise RuntimeError(f"Failed to save metadata: {str(e)}")

@celery.task(bind=True, name='app.process_video_task')
def process_video_task(self, video_path: str) -> str:
    """Celery task for video processing"""
    processing_id = str(uuid.uuid4())
    
    try:
        self.update_state(state='PROCESSING', meta={'step': 'Initializing'})
        
        processor = VideoProcessor(video_path, processing_id)
        processor.process_video(self)
        
        if os.path.exists(video_path):
            os.unlink(video_path)
        
        return processing_id

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        error_info = {
            'exc_type': type(e).__name__,
            'exc_message': str(e),
            'traceback': traceback.format_exc()
        }
        self.update_state(state=states.FAILURE, meta=error_info)
        raise Exception(error_info)

@worker_init.connect
def init_worker(**kwargs):
    """Initialize worker configuration"""
    Config.init_folders()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        video_id = str(uuid.uuid4())
        video_path = str(Config.UPLOAD_FOLDER / f"{video_id}.mp4")
        file.save(video_path)
        
        task = process_video_task.delay(video_path)
        return jsonify({
            "task_id": task.id,
            "video_id": video_id
        }), 202

    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({"error": "Upload failed"}), 500

@app.route('/status/<task_id>')
def get_status(task_id: str):
    """Get task status"""
    task = process_video_task.AsyncResult(task_id)
    
    try:
        response = {
            'state': task.state,
            'status': 'Pending'
        }
        
        if task.state == 'FAILURE':
            error_info = task.info
            response.update({
                'status': 'Failed',
                'error': error_info.get('exc_message', str(error_info))
            })
        elif task.state == 'SUCCESS':
            response.update({
                'status': 'Completed',
                'processing_id': task.result,
                'redirect_url': url_for('output', processing_id=task.result)
            })
        elif task.state in ['PROCESSING', 'PROGRESS']:
            response['status'] = task.info.get('step', 'Processing')
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Status check error: {str(e)}", exc_info=True)
        return jsonify({'state': 'ERROR', 'status': 'Error checking status'}), 500

@app.route('/output/<processing_id>')
def output(processing_id: str):
    """Display processing results"""
    try:
        metadata_path = Config.UPLOAD_FOLDER / processing_id / "metadata.json"
        if not metadata_path.exists():
            return render_template('error.html', error="Results not found"), 404
            
        with open(metadata_path, 'r') as f:
            chunks = json.load(f)
            
        return render_template('output.html', 
                             processing_id=processing_id,
                             chunks=chunks)
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}", exc_info=True)
        return render_template('error.html', error="Error displaying results"), 500

@app.route('/download/<processing_id>/<filename>')
def download_file(processing_id: str, filename: str):
    """Download processed file"""
    file_path = Config.UPLOAD_FOLDER / processing_id / filename
    
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        return send_file(str(file_path), as_attachment=True)
    except Exception as e:
        logger.error(f"Download error: {str(e)}", exc_info=True)
        return jsonify({"error": "Download failed"}), 500

if __name__ == '__main__':
    Config.init_folders()
    app.run(debug=True)