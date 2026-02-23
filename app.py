import os
import glob
import json
import streamlit as st
import subprocess
from datetime import datetime
from dotenv import load_dotenv

from add_channel import add_channel, get_channel_id_from_url, search_channel_by_name
from googleapiclient.discovery import build

# App Config
st.set_page_config(page_title="TheBrief Dashboard", page_icon="🎙️", layout="wide")
load_dotenv()

# --- Utility Functions ---

def load_briefs():
    """Load all markdown briefs from the briefs folder."""
    if not os.path.exists("briefs"):
        return []
        
    brief_files = glob.glob("briefs/*.md")
    brief_files.sort(reverse=True) # newest first
    
    briefs = []
    for f in brief_files:
        filename = os.path.basename(f)
        date_str = filename.replace(".md", "")
        with open(f, "r") as file:
            content = file.read()
            briefs.append({"date": date_str, "content": content, "path": f})
    return briefs

def load_channels():
    """Load the channels from channels.json"""
    channels_file = "channels.json"
    if not os.path.exists(channels_file):
        return []
    with open(channels_file, "r") as f:
        data = json.load(f)
        return data.get("channels", [])

def save_channels(channels):
    """Save the channels to channels.json"""
    with open("channels.json", "w") as f:
        json.dump({"channels": channels}, f, indent=2)

def load_queue():
    """Load the current video queue"""
    queue_file = os.path.join("data", "queue.json")
    if not os.path.exists(queue_file):
        return []
    try:
        with open(queue_file, "r") as f:
            return json.load(f)
    except:
        return []

# --- Custom Styling ---
st.markdown("""
    <style>
    .stApp header {background-color: transparent;}
    .brief-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    .channel-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        border-bottom: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)


# --- UI Layout ---

st.title("🎙️ TheBrief Dashboard")
st.markdown("Your daily deep-dive podcast briefing system.")

tab1, tab2, tab3 = st.tabs(["📑 Daily Briefs", "📺 Channels", "⚙️ Pipeline & Queue"])


# === TAB 1: Daily Briefs ===
with tab1:
    st.header("Latest Briefs")
    briefs = load_briefs()
    
    if not briefs:
        st.info("No briefs generated yet. Run the pipeline to generate your first brief!")
    else:
        # Sidebar selection for dates
        selected_date = st.selectbox(
            "Select Date", 
            options=[b["date"] for b in briefs],
            index=0
        )
        
        # Display selected brief
        selected_brief = next((b for b in briefs if b["date"] == selected_date), None)
        if selected_brief:
            st.markdown(f"🗓️ **Briefing for {selected_brief['date']}**")
            st.divider()
            
            # Use Streamlit's native markdown rendering which supports all our Gitea/Github formatting
            st.markdown(selected_brief["content"], unsafe_allow_html=True)


# === TAB 2: Channels ===
with tab2:
    st.header("Tracked Channels")
    channels = load_channels()
    
    # Add new channel form
    with st.expander("➕ Add New Channel", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_channel_input = st.text_input("YouTube Channel Name, URL, or Handle", placeholder="e.g. Lex Fridman or @LexFridman")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True) # spacing
            if st.button("Search & Add", use_container_width=True):
                if new_channel_input:
                    with st.spinner(f"Searching for '{new_channel_input}'..."):
                        youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
                        c_id, c_name = None, None
                        
                        if "youtube.com" in new_channel_input or "youtu.be" in new_channel_input:
                            c_id, c_name = get_channel_id_from_url(youtube, new_channel_input)
                        else:
                            c_id, c_name = search_channel_by_name(youtube, new_channel_input)
                            
                        if not c_id:
                            st.error(f"Could not find a channel matching '{new_channel_input}'")
                        else:
                            # Fetch name if missing
                            if not c_name:
                                resp = youtube.channels().list(part="snippet", id=c_id).execute()
                                c_name = resp["items"][0]["snippet"]["title"] if resp.get("items") else "Unknown"
                                
                            if any(c.get("id") == c_id for c in channels):
                                st.warning(f"Channel '{c_name}' is already being tracked.")
                            else:
                                channels.append({"name": c_name, "id": c_id})
                                save_channels(channels)
                                st.success(f"Added {c_name}!")
                                st.rerun()

    st.divider()
    
    # Display existing channels with delete buttons
    if not channels:
        st.info("No channels are currently being tracked.")
    else:
        for i, channel in enumerate(channels):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{channel['name']}**  \n`{channel['id']}`")
            with col2:
                if st.button("Remove", key=f"del_{i}", type="secondary", use_container_width=True):
                    channels.pop(i)
                    save_channels(channels)
                    st.rerun()
            st.divider()


# === TAB 3: Pipeline & Queue ===
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Pipeline Execution")
        st.markdown("The pipeline runs automatically every day at 6AM UTC via GitHub actions. You can also trigger it manually here.")
        
        if st.button("▶️ Run Pipeline Now", type="primary", use_container_width=True):
            with st.status("Running TheBrief Pipeline...", expanded=True) as status:
                st.write("Initializing...")
                try:
                    # Run main.py as a subprocess and capture output
                    # Using Popen to stream output if we want, but for simplicity here we just run it blockingly
                    process = subprocess.Popen(
                        ["python", "main.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    
                    # Read output line by line
                    for line in iter(process.stdout.readline, ""):
                        if line.strip():
                            st.text(line.strip())
                            
                    process.stdout.close()
                    return_code = process.wait()
                    
                    if return_code == 0:
                        status.update(label="Pipeline completed successfully!", state="complete", expanded=False)
                        st.balloons()
                        # Little hack to wait a sec then refresh the UI
                        st.rerun()
                    else:
                        status.update(label="Pipeline failed.", state="error", expanded=True)
                except Exception as e:
                    st.error(f"Error running pipeline: {e}")
                    status.update(label="Pipeline failed.", state="error")

    with col2:
        st.header("Current Queue")
        queue = load_queue()
        
        if not queue:
            st.info("Queue is empty. No unprocessed videos discovered.")
        else:
            st.warning(f"There are {len(queue)} videos in the queue.")
            for i, video in enumerate(queue):
                with st.expander(f"📺 {video.get('title', 'Unknown')}", expanded=(i==0)):
                    st.markdown(f"**Channel:** {video.get('channel', 'Unknown')}")
                    
                    if video.get('duration_minutes'):
                        st.markdown(f"**Duration:** {video.get('duration_minutes'):.1f} mins")
                        
                    st.markdown(f"[Watch on YouTube]({video.get('url', '#')})")
                    
                    # Status indicators based on keys
                    status = "Pending Download"
                    if video.get('audio_path'):
                        status = "Downloaded (Pending Transcription)"
                    if video.get('transcript'):
                        status = "Transcribed (Pending Summarization)"
                        
                    st.markdown(f"**Status:** `{status}`")
