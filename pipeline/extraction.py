import os
import subprocess
import json
import logging
import sys
import time
import random
import shutil

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds


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
        
    return "ffmpeg"  # Final fallback


def _ensure_cookies_file():
    """Ensure cookies.txt exists if cookie data is available. Call once before extraction loop."""
    if os.path.exists("cookies.txt"):
        return
    if "YOUTUBE_COOKIES" in os.environ:
        with open("cookies.txt", "w") as f:
            f.write(os.environ["YOUTUBE_COOKIES"])
        logger.info("cookies.txt created from YOUTUBE_COOKIES env var")


def _log_environment_diagnostics():
    """Log yt-dlp version and plugin availability for debugging CI issues."""
    try:
        ver = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            capture_output=True, text=True, timeout=10
        )
        logger.info(f"yt-dlp version: {ver.stdout.strip()}")
    except Exception as e:
        logger.warning(f"Could not get yt-dlp version: {e}")

    # Check if bgutil plugin is discoverable in the current Python env
    try:
        pkgs = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=columns"],
            capture_output=True, text=True, timeout=10
        )
        pot_packages = [line for line in pkgs.stdout.splitlines() if "pot" in line.lower() or "bgutil" in line.lower()]
        if pot_packages:
            logger.info(f"POT-related packages: {', '.join(p.strip() for p in pot_packages)}")
        else:
            logger.warning("No POT provider packages found in current Python environment")
    except Exception as e:
        logger.warning(f"Could not list packages: {e}")

    # Check if bgutil HTTP server is reachable
    try:
        import urllib.request
        req = urllib.request.urlopen("http://127.0.0.1:4416/", timeout=2)
        req.close()
        logger.info("bgutil POT provider HTTP server detected on port 4416")
    except Exception:
        logger.info("bgutil HTTP server not available on port 4416 (will rely on cookies)")


def extract_audio_for_video(video):
    """Extracts audio for a single video using yt-dlp with retry logic."""
    video_id = video["id"]
    video_url = video["url"]
    
    # We use the top-level audio directory
    os.makedirs("audio", exist_ok=True)
    output_template = os.path.join("audio", f"{video_id}.%(ext)s")
    final_output_path = os.path.join("audio", f"{video_id}.mp3")
    
    if os.path.exists(final_output_path):
        video["audio_path"] = final_output_path
        return video

    ffmpeg_path = get_ffmpeg_path()

    # Detect node.js runtime for yt-dlp n-challenge solver
    # yt-dlp 2026.x requires a JS runtime + remote solver script to bypass YouTube throttling
    node_path = shutil.which("node") or "/opt/homebrew/bin/node" or "/usr/local/bin/node"
    
    command = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "--no-check-certificate",
        "--prefer-free-formats",
        "-x",
        "--audio-format", "mp3",
        "--postprocessor-args", "-ar 16000 -ac 1",
        "--ffmpeg-location", ffmpeg_path,
        "--force-ipv4",
        "--js-runtimes", f"node:{node_path}",
        "--remote-components", "ejs:github",
        "-o", output_template
    ]

    # Cookie authentication: prefer cookies.txt (CI), fall back to browser cookies (local)
    if os.path.exists("cookies.txt"):
        command.extend(["--cookies", "cookies.txt"])
    else:
        # Local runs: extract cookies directly from Chrome
        command.extend(["--cookies-from-browser", "chrome"])
    
    command.append(video_url)
    
    # Retry loop with exponential backoff
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Pre-request jitter to avoid burst patterns
            time.sleep(random.uniform(1, 4))
            
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            if os.path.exists(final_output_path):
                video["audio_path"] = final_output_path
                logger.info(f"Successfully extracted: {final_output_path}")
                return video
            else:
                logger.error(f"Output file {final_output_path} not found after extraction. Stderr: {result.stderr}")
                return None
        except subprocess.CalledProcessError as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 3)
            if attempt < MAX_RETRIES:
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES} failed for {video_url}. "
                    f"Retrying in {delay:.1f}s..."
                )
                if e.stderr:
                    logger.debug(f"yt-dlp stderr: {e.stderr[:500]}")
                time.sleep(delay)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for {video_url}: {e}")
                if e.stderr:
                    logger.error(f"yt-dlp stderr: {e.stderr}")
                if e.stdout:
                    logger.debug(f"yt-dlp stdout: {e.stdout}")
                return None
    
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

    # One-time setup: cookies + environment diagnostics (before the loop)
    _ensure_cookies_file()
    _log_environment_diagnostics()
    
    processed_queue = []
    start_time = time.time()
    
    for i, video in enumerate(queue):
        # Check if already exists
        output_path = os.path.join("audio", f"{video['id']}.mp3")
        if os.path.exists(output_path):
            video["audio_path"] = output_path
            processed_queue.append(video)
            continue
            
        logger.info(f"Extracting audio for {video['title']} ({video['id']})...")
        result = extract_audio_for_video(video)
        if result:
            processed_queue.append(result)

        # Inter-video pacing: 5-12s between downloads to avoid rate-limiting
        if i < len(queue) - 1:
            pacing_delay = random.uniform(5, 12)
            logger.debug(f"Pacing delay: {pacing_delay:.1f}s before next download")
            time.sleep(pacing_delay)
            
    # Update queue with audio paths
    with open(queue_path, "w") as f:
        json.dump(processed_queue, f, indent=2)
        
    elapsed = time.time() - start_time
    logger.info(f"Extraction complete. {len(processed_queue)}/{len(queue)} audio files ready. (Time: {elapsed:.1f}s)")
    
    # Fail-safe: If we had a queue but got 0 results, error out
    if queue and not processed_queue:
        logger.error("CRITICAL: Extraction phase resulted in 0 audio files despite having a queue. Possible YouTube Bot Block.")
        sys.exit(1)
        
    return processed_queue

if __name__ == "__main__":
    run_extraction()
