import os
import subprocess
import json
import logging
import concurrent.futures
import time
import random

logger = logging.getLogger(__name__)

def get_ffmpeg_path():
    """Locate the ffmpeg binary, prioritizing static-ffmpeg then system path."""
    try:
        import static_ffmpeg
        base = os.path.dirname(static_ffmpeg.__file__)
        # Platform-specific subdirectories used by static-ffmpeg
        platform_bins = [
            os.path.join(base, "bin", "darwin_arm64", "ffmpeg"),
            os.path.join(base, "bin", "linux_x64", "ffmpeg"),
            os.path.join(base, "bin", "win32_x64", "ffmpeg.exe")
        ]
        for path in platform_bins:
            if os.path.exists(path):
                return path
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
        video["audio_path"] = final_output_path
        return video
        
    # Discover yt-dlp path
    # Try venv first (local), then system path (GitHub Actions/Global)
    ytdlp_path = "yt-dlp"
    potential_venv = os.path.join(os.getcwd(), "venv", "bin", "yt-dlp")
    if os.path.exists(potential_venv):
        ytdlp_path = potential_venv

    ffmpeg_path = get_ffmpeg_path()
    
    command = [
        ytdlp_path,
        "-x",
        "--audio-format", "mp3",
        "--postprocessor-args", "-ar 16000 -ac 1",
        "--ffmpeg-location", ffmpeg_path,
        "--force-ipv4",
        "--extractor-args", "youtube:player_client=ios,android",
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
    start_time = time.time()
    
    for video in queue:
        # Check if already exists
        output_template = os.path.join("audio", f"{video['id']}.mp3")
        if os.path.exists(output_template):
            # Already exists
            video["audio_path"] = output_template
            processed_queue.append(video)
            continue
            
        logger.info(f"Extracting audio for {video['title']} ({video['id']})...")
        result = extract_audio_for_video(video)
        if result:
            processed_queue.append(result)
            
    # Update queue with audio paths
    with open(queue_path, "w") as f:
        json.dump(processed_queue, f, indent=2)
        
    elapsed = time.time() - start_time
    logger.info(f"Extraction complete. {len(processed_queue)} audio files ready. (Time: {elapsed:.1f}s)")
    return processed_queue

if __name__ == "__main__":
    run_extraction()
