import os
import subprocess
import json
import logging
import concurrent.futures
import time
import random

logger = logging.getLogger(__name__)

def get_ffmpeg_path():
    """Locate the ffmpeg binary from the local static-ffmpeg package."""
    # Check common locations in venv
    venv_site_packages = os.path.join(os.getcwd(), "venv", "lib", "python3.13", "site-packages")
    local_path = os.path.join(venv_site_packages, "static_ffmpeg", "bin", "darwin_arm64", "ffmpeg")
    
    if os.path.exists(local_path):
        return local_path
        
    # Fallback to searching if venv path is different or not in cwd
    try:
        import static_ffmpeg
        # static_ffmpeg doesn't have a direct "get_path" in some versions, 
        # but we can try to find it via its module location
        base = os.path.dirname(static_ffmpeg.__file__)
        search_path = os.path.join(base, "bin", "darwin_arm64", "ffmpeg")
        if os.path.exists(search_path):
            return search_path
    except ImportError:
        pass
        
    return "ffmpeg" # Fallback to system path

def extract_audio_for_video(video):
    """Extracts audio for a single video using yt-dlp."""
    # Introduce random jitter (1-4 seconds) to prevent YouTube bot detection when threaded
    time.sleep(random.uniform(1, 4))

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
    
    ffmpeg_path = get_ffmpeg_path()
    
    command = [
        os.path.join(os.getcwd(), "venv", "bin", "yt-dlp"), # Use venv path explicitly
        "-x",
        "--audio-format", "mp3",
        "--postprocessor-args", "-ar 16000 -ac 1",
        "--ffmpeg-location", ffmpeg_path,
        "-o", output_template,
        video_url
    ]
    
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
    
    # Run audio extraction concurrently with a lower worker pool to avoid 429s
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all extraction jobs
        future_to_video = {executor.submit(extract_audio_for_video, video): video for video in queue}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_video):
            result = future.result()
            if result:
                processed_queue.append(result)
            
    # Update queue with audio paths
    with open(queue_path, "w") as f:
        json.dump(processed_queue, f, indent=2)
        
    logger.info(f"Extraction complete. {len(processed_queue)} audio files ready.")
    return processed_queue

if __name__ == "__main__":
    run_extraction()
