import os
import subprocess
import json
import logging

logger = logging.getLogger(__name__)

def extract_audio_for_video(video):
    """Extracts audio for a single video using yt-dlp."""
    video_id = video["id"]
    video_url = video["url"]
    
    # We use the top-level audio directory
    os.makedirs("audio", exist_ok=True)
    output_template = os.path.join("audio", f"{video_id}.%(ext)s")
    final_output_path = os.path.join("audio", f"{video_id}.mp3")
    
    if os.path.exists(final_output_path):
        logger.info(f"Audio already exists for {video_id}, skipping extraction.")
        video["audio_path"] = final_output_path
        return video
        
    logger.info(f"Extracting audio for {video_id} ({video['title']})...")
    
    # ... (command definition)
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        if os.path.exists(final_output_path):
            video["audio_path"] = final_output_path
            logger.info(f"Successfully extracted: {final_output_path}")
            return video
        else:
            logger.error(f"Output file {final_output_path} not found after extraction.")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio for {video_url}: {e}")
        return None

def run_extraction():
    queue_path = os.path.join("data", "queue.json")
    if not os.path.exists(queue_path):
        logger.error("No queue found. Run discovery first.")
        return []
        
    with open(queue_path, "r") as f:
        queue = json.load(f)
        
    if not queue:
        logger.info("Queue is empty.")
        return []
        
    processed_queue = []
    for video in queue:
        result = extract_audio_for_video(video)
        if result:
            processed_queue.append(result)
            
    # Update queue with audio paths
    with open(queue_path, "w") as f:
        json.dump(processed_queue, f, indent=2)
        
    logger.info(f"Extraction complete. {len(processed_queue)} audio files ready.")
    return processed_queue

if __name__ == "__main__":
    run_extraction()
