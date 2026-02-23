import os
import json
import logging
from faster_whisper import WhisperModel
import openai

logger = logging.getLogger(__name__)

def transcribe_local(audio_path, model_size="medium"):
    """Transcribes audio using local faster-whisper model."""
    print(f"Transcribing locally with faster-whisper (model: {model_size})...")
    
    # device="auto" automatically uses MPS on Apple Silicon or CUDA if available
    model = WhisperModel(model_size, device="auto", compute_type="default")
    
    segments, info = model.transcribe(audio_path, language=None)
    
    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
    
    transcript = " ".join([segment.text for segment in segments])
    return transcript

def transcribe_api(audio_path):
    """Transcribes audio using OpenAI Whisper API."""
    print("Transcribing via OpenAI Whisper API...")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text

def run_transcription():
    queue_path = os.path.join("data", "queue.json")
    if not os.path.exists(queue_path):
        logger.error("No queue found. Run extraction first.")
        return []
        
    with open(queue_path, "r") as f:
        queue = json.load(f)
        
    mode = os.getenv("TRANSCRIPTION_MODE", "api").lower()
    local_model = os.getenv("WHISPER_MODEL", "medium")
    
    processed_queue = []
    
    for video in queue:
        audio_path = video.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"Audio file missing for '{video.get('title')}', skipping transcription.")
            continue
            
        logger.info(f"Transcribing: {video['title']}")
        
        try:
            if mode == "local":
                transcript_text = transcribe_local(audio_path, local_model)
            else:
                transcript_text = transcribe_api(audio_path)
                
            video["transcript"] = transcript_text
            processed_queue.append(video)
            
            # Delete audio file to conserve disk space per spec
            try:
                os.remove(audio_path)
                logger.info(f"Deleted temp audio file: {audio_path}")
            except OSError as e:
                logger.error(f"Error deleting {audio_path}: {e}")
                
        except Exception as e:
            logger.error(f"Error transcribing {video['title']}: {e}")
            
    # Update queue with transcript texts
    with open(queue_path, "w") as f:
        json.dump(processed_queue, f, indent=2)
        
    logger.info(f"Transcription complete. {len(processed_queue)} transcripts ready.")
    return processed_queue

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_transcription()
