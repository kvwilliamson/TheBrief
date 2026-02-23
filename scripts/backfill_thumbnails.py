import os
import json
from googleapiclient.discovery import build
from dotenv import load_dotenv

def backfill_thumbnails():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YOUTUBE_API_KEY is not set.")
        return
        
    youtube = build("youtube", "v3", developerKey=api_key)
    
    channels_file = "channels.json"
    if not os.path.exists(channels_file):
        print(f"Error: {channels_file} not found.")
        return
        
    with open(channels_file, "r") as f:
        data = json.load(f)
        
    channels = data.get("channels", [])
    updated = False
    
    for channel in channels:
        if "thumbnail" not in channel or not channel["thumbnail"]:
            print(f"Fetching thumbnail for {channel['name']} ({channel['id']})...")
            try:
                resp = youtube.channels().list(part="snippet", id=channel["id"]).execute()
                if resp.get("items"):
                    thumb_url = resp["items"][0]["snippet"]["thumbnails"]["default"]["url"]
                    channel["thumbnail"] = thumb_url
                    updated = True
                    print(f"✅ Found thumbnail: {thumb_url}")
                else:
                    print(f"❌ Could not find channel {channel['id']} on YouTube.")
            except Exception as e:
                print(f"❌ Error fetching for {channel['name']}: {e}")
                
    if updated:
        with open(channels_file, "w") as f:
            json.dump(data, f, indent=2)
        print("\nSuccessfully updated channels.json with thumbnails.")
    else:
        print("\nNo updates needed. All channels already have thumbnails.")

if __name__ == "__main__":
    backfill_thumbnails()
