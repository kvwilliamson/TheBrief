import os
import sys
import json
from googleapiclient.discovery import build
import urllib.parse
from dotenv import load_dotenv

def get_channel_id_from_url(youtube, url):
    """Attempt to extract channel ID or handle from a YouTube URL."""
    parsed = urllib.parse.urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p]
    
    if not path_parts:
        return None, None
        
    if path_parts[0] == 'channel' and len(path_parts) > 1:
        # It's an exact channel ID (e.g. youtube.com/channel/UC12345)
        return path_parts[1], None
        
    if path_parts[0].startswith('@'):
        # It's a handle (e.g. youtube.com/@LexFridman)
        handle = path_parts[0]
        try:
            resp = youtube.search().list(part="snippet", type="channel", q=handle, maxResults=1).execute()
            if resp.get("items"):
                snippet = resp["items"][0]["snippet"]
                return snippet["channelId"], snippet["title"], snippet["thumbnails"]["default"]["url"]
        except Exception as e:
            print(f"Error searching for handle {handle}: {e}")
            
    # As a fallback, search the whole URL string as a query
    return search_channel_by_name(youtube, url)

def search_channel_by_name(youtube, name):
    """Search for a channel ID using a text query."""
    try:
        resp = youtube.search().list(part="snippet", type="channel", q=name, maxResults=1).execute()
        if resp.get("items"):
            item = resp["items"][0]["snippet"]
            return item["channelId"], item["channelTitle"], item["thumbnails"]["default"]["url"]
    except Exception as e:
        print(f"Error searching for channel '{name}': {e}")
    return None, None, None

def add_channel():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YOUTUBE_API_KEY is not set in your .env file.")
        sys.exit(1)
        
    youtube = build("youtube", "v3", developerKey=api_key)
    channels_file = "channels.json"
    
    # Load existing channels
    data = {"channels": []}
    if os.path.exists(channels_file):
        try:
            with open(channels_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {channels_file} was invalid JSON, creating a fresh one.")
            
    print("YouTube Channel Adder for TheBrief")
    print("-" * 35)
    
    while True:
        query = input("\nEnter a YouTube Channel URL or Name (or press Enter to quit): ").strip()
        if not query:
            break
            
        channel_id, channel_name, channel_thumb = None, None, None
        
        # Try as URL first, fallback to name search
        if "youtube.com" in query or "youtu.be" in query:
            channel_id, channel_name, channel_thumb = get_channel_id_from_url(youtube, query)
        else:
            channel_id, channel_name, channel_thumb = search_channel_by_name(youtube, query)
            
        if not channel_id:
            print(f"❌ Could not find a channel matching: '{query}'")
            continue
            
        # If we got an ID but no name or thumb, fetch them
        if not channel_name or not channel_thumb:
            try:
                resp = youtube.channels().list(part="snippet", id=channel_id).execute()
                if resp.get("items"):
                    snippet = resp["items"][0]["snippet"]
                    channel_name = channel_name or snippet["title"]
                    channel_thumb = channel_thumb or snippet["thumbnails"]["default"]["url"]
                else:
                    channel_name = channel_name or "Unknown Channel"
                    channel_thumb = channel_thumb or ""
            except Exception:
                channel_name = channel_name or "Unknown Channel"
                channel_thumb = channel_thumb or ""
                
        # Check for duplicates
        if any(c.get("id") == channel_id for c in data["channels"]):
            print(f"⚠️ Channel '{channel_name}' ({channel_id}) is already in your list.")
            continue
            
        # Add and save
        new_channel = {"name": channel_name, "id": channel_id, "thumbnail": channel_thumb}
        data["channels"].append(new_channel)
        
        with open(channels_file, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Successfully added '{channel_name}' to {channels_file}!")

if __name__ == "__main__":
    add_channel()
