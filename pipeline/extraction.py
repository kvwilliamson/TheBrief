import os
import subprocess
import json
import logging
import concurrent.futures
import time
import random

logger = logging.getLogger(__name__)

import shutil

def get_ffmpeg_path():
    """Locate the ffmpeg binary, prioritizing static-ffmpeg then system path."""
    # 1. Try static-ffmpeg via module
    try:
        import static_ffmpeg
        base = os.path.dirname(static_ffmpeg.__file__)
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
        
    # 2. Try system PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
        
    return "ffmpeg" # Final fallback

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
        "-f", "bestaudio/best",
        "--no-check-certificate",
        "--prefer-free-formats",
        "--extractor-args", "youtube:player_client=tv,mweb",
        "-x",
        "--audio-format", "mp3",
        "--postprocessor-args", "-ar 16000 -ac 1",
        "--ffmpeg-location", ffmpeg_path,
        "--force-ipv4",
        "-o", output_template
    ]

    # Add cookies if cookies.txt exists
    if os.path.exists("cookies.txt"):
        command.extend(["--cookies", "cookies.txt"])
    elif "YOUTUBE_COOKIES" in os.environ:
        # Fallback for transient environments if file isn't created yet
        with open("cookies.txt", "w") as f:
            f.write(os.environ["YOUTUBE_COOKIES"])
        command.extend(["--cookies", "cookies.txt"])
    
    command.append(video_url)
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if os.path.exists(final_output_path):
            video["audio_path"] = final_output_path
            logger.info(f"Successfully extracted: {final_output_path}")
            return video
        else:
            logger.error(f"Output file {final_output_path} not found after extraction. Stderr: {result.stderr}")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio for {video_url}: {e}")
        if e.stderr:
            logger.error(f"yt-dlp stderr: {e.stderr}")
        if e.stdout:
            logger.debug(f"yt-dlp stdout: {e.stdout}")
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
    
    # Fail-safe: If we had a queue but got 0 results, wait and error out
    if queue and not processed_queue:
        logger.error("CRITICAL: Extraction phase resulted in 0 audio files despite having a queue. Possible YouTube Bot Block.")
        import sys
        sys.exit(1)
        
    return processed_queue

if __name__ == "__main__":
    run_extraction()
