from flask import Flask, request, render_template, send_file, jsonify
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
from pathlib import Path
from typing import List, Dict, Any
import logging
from pytube import YouTube
import re
from werkzeug.utils import secure_filename
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import huggingface_hub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    UPLOAD_FOLDER = Path('uploads')
    CELERY_BROKER_URL = 'redis://localhost:6379/0'
    RESULT_BACKEND = 'redis://localhost:6379/0'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}
    WHISPER_MODEL = "base"
    MAX_CHUNK_DURATION = 10  # seconds
    YOUTUBE_REGEX = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )

    @staticmethod
    def init_folders():
        """Initialize necessary folders"""
        folders = [
            Config.UPLOAD_FOLDER,
            Config.UPLOAD_FOLDER / "temp"
        ]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    @staticmethod
    def validate_youtube_url(url: str) -> bool:
        return bool(Config.YOUTUBE_REGEX.match(url))

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
    def __init__(self, video_path: str, processing_id: str, is_youtube: bool = False):
        self.video_path = video_path
        self.processing_id = processing_id
        self.output_dir = Config.UPLOAD_FOLDER / processing_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.is_youtube = is_youtube

    def _download_youtube_video(self):
        """Download YouTube video with error handling"""
        try:
            yt = YouTube(self.video_path)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            if not stream:
                raise ValueError("No suitable video stream found")
            
            self.video_path = str(self.output_dir / f"{self.processing_id}_yt.mp4")
            stream.download(filename=self.video_path)
        except Exception as e:
            logger.error(f"YouTube download failed: {str(e)}")
            raise RuntimeError(f"YouTube download failed: {str(e)}")

    def initialize_model(self):
        """Initialize Whisper model with error handling"""
        try:
            self.model = whisper.load_model(Config.WHISPER_MODEL, device=self.device)
        except Exception as e:
            logger.error(f"Model load error: {str(e)}")
            raise

    def process_video(self, task) -> List[Dict[str, Any]]:
        """Main processing pipeline with YouTube support"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.is_youtube:
                task.update_state(state='PROCESSING', meta={'step': 'Downloading YouTube video'})
                self._download_youtube_video()

            task.update_state(state='PROCESSING', meta={'step': 'Extracting audio'})
            transcript = self._transcribe_audio()

            task.update_state(state='PROCESSING', meta={'step': 'Processing chunks'})
            chunks = self._process_chunks(transcript)

            self._save_metadata(chunks)
            return chunks

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise

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
def process_video_task(self, video_path: str, is_youtube: bool = False) -> str:
    """Enhanced Celery task with YouTube support"""
    processing_id = str(uuid.uuid4())
    
    try:
        processor = VideoProcessor(video_path, processing_id, is_youtube)
        processor.process_video(self)
        
        if not is_youtube and os.path.exists(video_path):
            os.unlink(video_path)
            
        return processing_id

    except Exception as e:
        logger.error(f"Task error: {str(e)}", exc_info=True)
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'traceback': traceback.format_exc()
            }
        )
        raise Ignore()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload or YouTube URL with proper task argument handling"""
    try:
        if 'video' not in request.files and 'youtube_url' not in request.form:
            return jsonify({"error": "No file or YouTube URL provided"}), 400

        if 'video' in request.files:
            file = request.files['video']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            if not Config.validate_file_extension(file.filename):
                return jsonify({"error": "Invalid file type"}), 400

            filename = secure_filename(file.filename)
            file_path = str(Config.UPLOAD_FOLDER / filename)
            file.save(file_path)

            task = process_video_task.apply_async((file_path,), countdown=10)
            return jsonify({"task_id": task.id})

        elif 'youtube_url' in request.form:
            youtube_url = request.form['youtube_url']
            if not Config.validate_youtube_url(youtube_url):
                return jsonify({"error": "Invalid YouTube URL"}), 400

            task = process_video_task.apply_async((youtube_url, True), countdown=10)
            return jsonify({"task_id": task.id})

    except Exception as e:
        logger.error(f"Error in upload_video: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred during the video upload."}), 500

@app.route('/status/<task_id>', methods=['GET'])
def task_status(task_id):
    """Get the status of the video processing task"""
    task = process_video_task.AsyncResult(task_id)
    if task.state == states.PENDING:
        response = {'state': task.state, 'status': 'Waiting in queue'}
    elif task.state != states.FAILURE:
        response = {'state': task.state, 'status': task.info.get('step', '')}
    else:
        response = {'state': task.state, 'status': str(task.info)}
    
    return jsonify(response)

@app.route('/result/<task_id>', methods=['GET'])
def get_task_result(task_id):
    """Return the processed result of the video"""
    task = process_video_task.AsyncResult(task_id)
    if task.state == states.SUCCESS:
        return jsonify({"result": task.result})
    else:
        return jsonify({"error": "Task not finished yet or failed"}), 400

if __name__ == "__main__":
    Config.init_folders()
    app.run(debug=True)
