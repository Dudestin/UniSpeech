# WavLM Embedding API Server

This REST API server provides endpoints for extracting speaker embeddings from audio files using the WavLM Large model.

## Features

- **Audio Format Support**: Accepts MP3, M4A, WAV, and other common audio formats
- **Batch Processing**: Process multiple audio files in a single request
- **Safety Features**: 
  - File size limit (100MB default)
  - Audio duration clipping (30 seconds default)
  - Automatic temporary file cleanup
- **Automatic Preprocessing**:
  - Converts to 16kHz sampling rate
  - Converts to mono channel
  - Normalizes embeddings (L2 normalization)

## API Endpoints

### Health Check
```
GET /health
```
Returns server status and model loading state.

### Single Audio Embedding
```
POST /embed
```
Compute embedding for a single audio file.

**Request Body:**
```json
{
  "url": "https://example.com/audio.mp3",
  "max_duration_seconds": 30.0  // optional, default: 30.0
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, ...],  // 256-dimensional normalized vector
  "audio_duration": 5.2,
  "sample_rate": 16000
}
```

### Batch Audio Embedding
```
POST /embed_batch
```
Compute embeddings for multiple audio files.

**Request Body:**
```json
{
  "urls": [
    "https://example.com/audio1.mp3",
    "https://example.com/audio2.wav"
  ],
  "max_duration_seconds": 30.0  // optional, default: 30.0
}
```

**Response:**
```json
{
  "embeddings": [
    {
      "embedding": [0.123, -0.456, ...],
      "audio_duration": 5.2,
      "sample_rate": 16000
    },
    {
      "embedding": [0.789, -0.012, ...],
      "audio_duration": 3.8,
      "sample_rate": 16000
    }
  ]
}
```

## Configuration

### Environment Variables

- `PORT`: Server port (default: 8080)
- `CHECKPOINT_PATH`: Path to model checkpoint file (default: "wavlm_large_finetune.pth")

### Running the Server

```bash
python api_server.py
```

Or with custom port:
```bash
PORT=8000 python api_server.py
```

## Docker Deployment

The server is designed to run in a Docker container with the provided Dockerfile in the UniSpeech-docker repository.

## Error Handling

- **413 Payload Too Large**: Audio file exceeds size limit
- **400 Bad Request**: Failed to download audio from URL
- **500 Internal Server Error**: Processing error

Failed audio files in batch requests return empty embeddings with zero duration and sample rate.

## Technical Details

- **Model**: WavLM Large with ECAPA-TDNN backend
- **Embedding Dimension**: 256
- **Input Requirements**: 16kHz mono audio
- **Framework**: FastAPI with PyTorch backend
- **GPU Support**: Automatically uses GPU if available