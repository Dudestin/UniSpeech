import os
import tempfile
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import requests
from torchaudio.transforms import Resample
import librosa
from models.ecapa_tdnn import ECAPA_TDNN_SMALL
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="WavLM Embedding API", version="1.0.0")

# Global model instance
model = None
device = None

# Request/Response models
class AudioRequest(BaseModel):
    url: str
    max_duration_seconds: Optional[float] = 30.0

class BatchAudioRequest(BaseModel):
    urls: List[str]
    max_duration_seconds: Optional[float] = 30.0

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    audio_duration: float
    sample_rate: int

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[EmbeddingResponse]


def init_model():
    """Initialize the WavLM model"""
    global model, device
    
    # Check if GPU is available
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "wavlm_large_finetune.pth")
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict['model'], strict=False)
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, using default weights")
    
    model = model.to(device)
    model.eval()
    logger.info("Model initialized successfully")


def download_audio(url: str, max_size_mb: int = 100) -> bytes:
    """Download audio file from URL with size limit"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Audio file too large (max {max_size_mb}MB)")
        
        # Download with streaming
        chunks = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            chunks.append(chunk)
            downloaded += len(chunk)
            if downloaded > max_size_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"Audio file too large (max {max_size_mb}MB)")
        
        return b''.join(chunks)
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000, max_duration: float = 30.0) -> tuple:
    """Load audio from bytes and convert to 16kHz mono"""
    with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Try soundfile first (for wav files)
        try:
            audio, sr = sf.read(tmp_path)
        except:
            # Fall back to librosa for other formats (mp3, m4a, etc.)
            audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Clip audio if too long
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            logger.info(f"Clipping audio from {len(audio)/sr:.2f}s to {max_duration}s")
            audio = audio[:max_samples]
        
        # Resample to 16kHz if needed
        if sr != target_sr:
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            resampler = Resample(orig_freq=sr, new_freq=target_sr)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()
            sr = target_sr
        
        return audio, sr
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def compute_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute embedding vector from audio"""
    # Convert to tensor and add batch dimension
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
    audio_tensor = audio_tensor.to(device)
    
    # Compute embedding
    with torch.no_grad():
        embedding = model(audio_tensor)
    
    # Normalize embedding
    embedding = F.normalize(embedding, p=2, dim=1)
    
    return embedding.cpu().numpy().squeeze()


# Model is initialized before server startup in main()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_audio(request: AudioRequest):
    """Compute embedding for a single audio file"""
    try:
        # Download audio
        logger.info(f"Downloading audio from {request.url}")
        audio_bytes = download_audio(request.url)
        
        # Load and preprocess audio
        audio, sr = load_audio_from_bytes(audio_bytes, max_duration=request.max_duration_seconds)
        duration = len(audio) / sr
        
        # Compute embedding
        embedding = compute_embedding(audio, sr)
        
        return EmbeddingResponse(
            embedding=embedding.tolist(),
            audio_duration=duration,
            sample_rate=sr
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/embed_batch", response_model=BatchEmbeddingResponse)
async def embed_audio_batch(request: BatchAudioRequest):
    """Compute embeddings for multiple audio files"""
    embeddings = []
    
    for url in request.urls:
        try:
            # Download audio
            logger.info(f"Processing audio from {url}")
            audio_bytes = download_audio(url)
            
            # Load and preprocess audio
            audio, sr = load_audio_from_bytes(audio_bytes, max_duration=request.max_duration_seconds)
            duration = len(audio) / sr
            
            # Compute embedding
            embedding = compute_embedding(audio, sr)
            
            embeddings.append(EmbeddingResponse(
                embedding=embedding.tolist(),
                audio_duration=duration,
                sample_rate=sr
            ))
        except Exception as e:
            logger.error(f"Error processing audio from {url}: {str(e)}")
            # Add None placeholder for failed audio
            embeddings.append(EmbeddingResponse(
                embedding=[],
                audio_duration=0.0,
                sample_rate=0
            ))
    
    return BatchEmbeddingResponse(embeddings=embeddings)


if __name__ == "__main__":
    # Get port from environment variable (for Cloud Run)
    port = int(os.environ.get("PORT", 8080))
    
    # Initialize model before starting server to avoid startup timeout
    logger.info("Initializing model before starting server...")
    init_model()
    logger.info("Model initialization complete, starting server...")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)