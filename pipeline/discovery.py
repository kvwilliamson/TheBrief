import os
import json
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
import isodate
from tinydb import TinyDB, Query

def get_recent_videos(youtube, channel_id, published_after):
    """Fetch videos from a channel published after a certain date."""
    try:
        channels_response = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        ).execute()
        
        if not channels_response.get("items"):
            print(f"Channel not found: {channel_id}")
            return []
            
        uploads_playlist_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        
        videos = []
        next_page_token = None
        
        while True:
            playlist_response = youtube.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            
            items = playlist_response.get("items", [])
            if not items:
                break
                
            for item in items:
                pub_date_str = item["snippet"]["publishedAt"]
                pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                
                if pub_date >= published_after:
                    videos.append({
                        "id": item["snippet"]["resourceId"]["videoId"],
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"]["channelTitle"],
                        "channel_id": channel_id,
                        "published_at": pub_date_str
                    })
            
            next_page_token = playlist_response.get("nextPageToken")
            if not next_page_token:
                break
                
            # Stop pagination if the last item is older than the threshold
            last_pub_date_str = items[-1]["snippet"]["publishedAt"]
            last_pub_date = datetime.fromisoformat(last_pub_date_str.replace('Z', '+00:00'))
            if last_pub_date < published_after:
                break
                
        return videos
    except Exception as e:
        print(f"Error fetching for channel {channel_id}: {e}")
        return []

def filter_long_form(youtube, videos):
    """Filter videos to only those longer than 20 minutes."""
    if not videos:
        return []
        
    video_ids = [v["id"] for v in videos]
    long_videos = []
    
    # YouTube API allows max 50 ids per request
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        try:
            video_response = youtube.videos().list(
                part="contentDetails",
                id=",".join(batch_ids)
            ).execute()
            
            for item in video_response.get("items", []):
                duration_iso = item["contentDetails"]["duration"]
                try:
                    duration = isodate.parse_duration(duration_iso)
                    duration_mins = duration.total_seconds() / 60
                    
                    if duration_mins > 20: 
                        original_video = next(v for v in videos if v["id"] == item["id"])
                        # Add duration info
                        original_video["duration_iso"] = duration_iso
                        original_video["duration_minutes"] = duration_mins
                        original_video["url"] = f"https://www.youtube.com/watch?v={item['id']}"
                        long_videos.append(original_video)
                except Exception as e:
                    print(f"Error parsing duration {duration_iso} for video {item['id']}: {e}")
        except Exception as e:
            print(f"Error fetching details for batch: {e}")
            
    return long_videos

def run_discovery():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Warning: YOUTUBE_API_KEY not found in environment.")
        return []
        
    youtube = build("youtube", "v3", developerKey=api_key)
    
    channels_file = "channels.json"
    if not os.path.exists(channels_file):
        print(f"Error: {channels_file} not found.")
        return []
        
    with open(channels_file, "r") as f:
        channels_data = json.load(f)
        
    channels = channels_data.get("channels", [])
    if not channels:
        print("No channels configured.")
        return []
    
    # Threshold: X hours ago (default 24)
    lookback_hours = int(os.getenv("DISCOVERY_LOOKBACK_HOURS", "24"))
    published_after = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    print(f"Discovering videos published after {published_after.isoformat()} ({lookback_hours}h lookback)")
    
    all_recent_videos = []
    for channel in channels:
        print(f"Checking channel: {channel['name']} ({channel['id']})")
        recent = get_recent_videos(youtube, channel["id"], published_after)
        print(f"  Found {len(recent)} recent videos.")
        all_recent_videos.extend(recent)
        
    # Filter by duration > 20 mins
    long_videos = filter_long_form(youtube, all_recent_videos)
    print(f"Found {len(long_videos)} total long-form videos across all channels.")
    
    # Deduplicate against processed videos
    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", "processed_videos.json")
    db = TinyDB(db_path)
    Video = Query()
    
    queue = []
    for video in long_videos:
        if not db.contains(Video.id == video["id"]):
            queue.append(video)
        else:
            print(f"Skipping previously processed video: {video['title']}")
            
    # Save queue for next phase
    queue_path = os.path.join("data", "queue.json")
    with open(queue_path, "w") as f:
        json.dump(queue, f, indent=2)
        
    print(f"Discovery complete. Added {len(queue)} new videos to queue.")
    return queue

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_discovery()
