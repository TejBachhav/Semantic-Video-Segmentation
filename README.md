# Semantic-Video-Segmentation
---

# Video Processing Application

This application is designed to process video files, extract audio, and transcribe the content using OpenAI's Whisper model. It leverages Celery for asynchronous task management and provides an API for uploading videos, checking task status, and downloading transcribed audio chunks.

## Features

- **Video Processing**:
  - Upload and process video files (MP4, AVI, MOV, WMV).
  - Extract audio and transcribe it using OpenAI's Whisper model.
  - Split audio into semantic chunks for efficient processing.

- **Task Management**:
  - Asynchronous task processing using Celery.
  - Real-time task status updates.

- **Metadata and Downloads**:
  - Save and retrieve metadata for processed videos.
  - Download individual audio chunks.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.8 or higher
- Redis (for Celery task queue)
- FFmpeg (for audio processing)

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/video-processing-app.git
   cd video-processing-app
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Redis**:
   Ensure Redis is running on your system. You can install and start it using:
   ```bash
   sudo apt-get install redis-server
   redis-server
   ```

5. **Run the Application**:
   Start the Flask application and Celery worker:
   ```bash
   # Start Flask app
   python app.py

   # Start Celery worker (in a separate terminal)
   celery -A app.celery worker --loglevel=info
   ```

6. **Access the Application**:
   Open your browser and navigate to `http://localhost:5000`.

## Usage

### Uploading and Processing Videos

1. **Upload a Video File**:
   - Go to the homepage.
   - Select a video file and click "Upload".
   - The application will process the video and return a `video_id`.

### Task Status

- Use the `/status/<task_id>` endpoint to check the status of a processing task.

### Downloading Audio Chunks

- Use the `/download/<video_id>/<chunk_id>` endpoint to download specific audio chunks.

### Retrieving Metadata

- Use the `/metadata/<video_id>` endpoint to get metadata about the processed video.

## API Endpoints

| Endpoint                          | Method | Description                                      |
|-----------------------------------|--------|--------------------------------------------------|
| `/`                               | GET    | Homepage for uploading videos.                   |
| `/upload`                         | POST   | Upload a video file for processing.              |
| `/status/<task_id>`               | GET    | Check the status of a processing task.           |
| `/metadata/<video_id>`            | GET    | Retrieve metadata for a processed video.         |
| `/download/<video_id>/<chunk_id>` | GET    | Download a specific audio chunk.                 |

## Configuration

The application can be configured using the `Config` class in `app.py`. Key configuration options include:

- `UPLOAD_FOLDER`: Directory for storing uploaded files.
- `CELERY_BROKER_URL`: URL for the Celery broker (Redis).
- `WHISPER_MODEL`: Whisper model to use for transcription.
- `MAX_CONTENT_LENGTH`: Maximum file size for uploads (default: 500MB).
- `ALLOWED_EXTENSIONS`: List of allowed video file extensions.

## Troubleshooting

- **File Upload Issues**:
  - Ensure the file size is within the limit (500MB).
  - Check that the file extension is allowed.

- **Task Failures**:
  - Check the Celery worker logs for detailed error messages.
  - Ensure Redis is running and accessible.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This updated `README.md` focuses on the core functionality of the application, which is video file processing and audio transcription. It provides clear instructions for setup, usage, and troubleshooting. Let me know if you need further adjustments!Here’s the updated `README.md` file after removing the **RAG (Retrieval-Augmented Generation)** and **YouTube link processing** parts. The application now focuses solely on processing uploaded video files, extracting audio, and transcribing the content.

---
