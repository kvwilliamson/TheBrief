import os
import json
import logging
import concurrent.futures
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
import isodate
from tinydb import TinyDB, Query

logger = logging.getLogger(__name__)

def get_recent_videos(youtube, channel_id, published_after):
    """Fetch videos from a channel published after a certain date."""
    try:
        channels_response = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        ).execute()
        
        if not channels_response.get("items"):
            logger.warning(f"Channel not found: {channel_id}")
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
                        "thumbnail": item["snippet"]["thumbnails"].get("high", {}).get("url", item["snippet"]["thumbnails"].get("default", {}).get("url", "")),
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
        logger.error(f"Error fetching for channel {channel_id}: {e}")
        return []

def filter_long_form(youtube, videos):
    """Filter videos to only those longer than 5 minutes."""
    if not videos:
        return []
        
    video_ids = [v["id"] for v in videos]
    long_videos = []
    
    # YouTube API allows max 50 ids per request
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        try:
            video_response = youtube.videos().list(
                part="contentDetails,snippet",
                id=",".join(batch_ids)
            ).execute()
            
            for item in video_response.get("items", []):
                content_details = item.get("contentDetails", {})
                duration_iso = content_details.get("duration")
                if not duration_iso:
                    continue # Skip premiere or live stream without a set duration
                    
                try:
                    duration = isodate.parse_duration(duration_iso)
                    duration_mins = duration.total_seconds() / 60
                    
                    if duration_mins > 5: 
                        original_video = next((v for v in videos if v["id"] == item["id"]), None)
                        if original_video:
                            # Add duration info
                            original_video["duration_iso"] = duration_iso
                            original_video["duration_minutes"] = duration_mins
                            original_video["url"] = f"https://www.youtube.com/watch?v={item['id']}"
                            original_video["tags"] = item["snippet"].get("tags", [])
                            long_videos.append(original_video)
                except Exception as e:
                    logger.error(f"Error parsing duration {duration_iso} for video {item['id']}: {e}")
        except Exception as e:
            logger.error(f"Error fetching details for batch: {e}")
            
    return long_videos

def run_discovery():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.warning("YOUTUBE_API_KEY not found in environment.")
        return []
        
    youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    
    channels_file = "channels.json"
    if not os.path.exists(channels_file):
        logger.error(f"{channels_file} not found.")
        return []
        
    with open(channels_file, "r") as f:
        channels_data = json.load(f)
        
    channels = channels_data.get("channels", [])
    if not channels:
        logger.info("No channels configured.")
        return []
    
    lookback_hours = int(os.getenv("DISCOVERY_LOOKBACK_HOURS", "24"))
    published_after = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    logger.info(f"Discovering videos published after {published_after.isoformat()} ({lookback_hours}h lookback)")
    
    all_recent_videos = []
    
    def process_channel(channel):
        # logger.debug(f"Checking {channel['name']}...")
        local_youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        recent = get_recent_videos(local_youtube, channel["id"], published_after)
        if recent:
            logger.info(f"Found {len(recent)} new potential videos on {channel['name']}.")
            for v in recent:
                v["category"] = channel.get("category", "Other")
        return recent

    # Run YouTube API requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(process_channel, channels)
        for res in results:
            all_recent_videos.extend(res)
        
    # Filter by duration > 5 mins
    long_videos = filter_long_form(youtube, all_recent_videos)
    logger.info(f"Found {len(long_videos)} total long-form videos across all channels.")
    
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
            logger.debug(f"Skipping previously processed video: {video['title']}")
    # Sort queue by published date (newest first) as a logical default
    queue.sort(key=lambda x: x.get("published_at", ""), reverse=True)

    # Save queue for next phase
    queue_path = os.path.join("data", "queue.json")
    with open(queue_path, "w") as f:
        json.dump(queue, f, indent=2)
        
    logger.info(f"Discovery complete. Added {len(queue)} new videos to queue.")
    return queue

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_discovery()
