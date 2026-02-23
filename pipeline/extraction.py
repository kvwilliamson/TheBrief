import os
import subprocess
import json

def extract_audio_for_video(video):
    """Extracts audio for a single video using yt-dlp."""
    video_id = video["id"]
    video_url = video["url"]
    
    # We use the top-level audio directory
    os.makedirs("audio", exist_ok=True)
    output_template = os.path.join("audio", f"{video_id}.%(ext)s")
    final_output_path = os.path.join("audio", f"{video_id}.mp3")
    
    if os.path.exists(final_output_path):
        print(f"Audio already exists for {video_id}, skipping extraction.")
        video["audio_path"] = final_output_path
        return video
        
    print(f"Extracting audio for {video_id} ({video['title']})...")
    
    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--postprocessor-args", "-ar 16000 -ac 1",
        "-o", output_template,
        video_url
    ]
    
    try:
        subprocess.run(command, check=True)
        if os.path.exists(final_output_path):
            video["audio_path"] = final_output_path
            print(f"Successfully extracted: {final_output_path}")
            return video
        else:
            print(f"Error: Output file {final_output_path} not found after extraction.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio for {video_url}: {e}")
        return None

def run_extraction():
    queue_path = os.path.join("data", "queue.json")
    if not os.path.exists(queue_path):
        print("No queue found. Run discovery first.")
        return []
        
    with open(queue_path, "r") as f:
        queue = json.load(f)
        
    if not queue:
        print("Queue is empty.")
        return []
        
    processed_queue = []
    for video in queue:
        result = extract_audio_for_video(video)
        if result:
            processed_queue.append(result)
            
    # Update queue with audio paths
    with open(queue_path, "w") as f:
        json.dump(processed_queue, f, indent=2)
        
    print(f"Extraction complete. {len(processed_queue)} audio files ready.")
    return processed_queue

if __name__ == "__main__":
    run_extraction()
