import os
import glob
import json
import streamlit as st
import subprocess
import requests
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

@st.cache_data(ttl=3600)
def get_image_bytes(url):
    """Fetch image bytes on the backend to avoid browser console errors."""
    if not url:
        return None
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return response.content
    except:
        pass
    return None

def show_channel_image(url):
    """Safely render a channel image by fetching bytes on the backend."""
    img_bytes = get_image_bytes(url)
    if img_bytes:
        st.image(img_bytes, width=80)
    else:
        st.markdown("<div style='width:80px;height:80px;display:flex;align-items:center;justify-content:center;background:#333;border-radius:50%;font-size:30px;'>📻</div>", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'channels' not in st.session_state:
    st.session_state.channels = load_channels()

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

def refresh_channels():
    st.session_state.channels = load_channels()

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

tab1, tab2, tab3, tab4 = st.tabs(["📑 Daily Briefs", "📺 Channels", "🔍 Discover", "⚙️ Pipeline & Queue"])


# === TAB 1: Daily Briefs ===
with tab1:
    st.header("Latest Briefs")
    briefs = load_briefs()
    queue = load_queue()
    
    # --- Status Overview Card ---
    last_brief_date = briefs[0]["date"] if briefs else "Never"
    pending_count = len(queue)
    
    st.markdown(f"""
        <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 5px solid #00d1b2; margin-bottom: 25px; color: #ffffff;'>
            <h3 style='margin-top: 0; color: #ffffff;'>Welcome Back! 👋</h3>
            <p style='margin-bottom: 5px;'>📅 <b>Last Brief Generated:</b> {last_brief_date}</p>
            <p style='margin-bottom: 10px;'>📹 <b>Pending in Queue:</b> {pending_count} new video{'s' if pending_count != 1 else ''} ready to be briefed.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("✨ Check for New Videos", use_container_width=True):
            with st.spinner("Scanning channels..."):
                try:
                    from pipeline.discovery import run_discovery
                    new_videos = run_discovery()
                    st.success(f"Discovered {len(new_videos)} new videos!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Discovery failed: {e}")

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
                            c_id, c_name, c_thumb = get_channel_id_from_url(youtube, new_channel_input)
                        else:
                            c_id, c_name, c_thumb = search_channel_by_name(youtube, new_channel_input)
                            
                        if not c_id:
                            st.error(f"Could not find a channel matching '{new_channel_input}'")
                        else:
                            # Fetch name/thumb if missing
                            if not c_name or not c_thumb:
                                resp = youtube.channels().list(part="snippet", id=c_id).execute()
                                if resp.get("items"):
                                    snippet = resp["items"][0]["snippet"]
                                    c_name = c_name or snippet["title"]
                                    c_thumb = c_thumb or snippet["thumbnails"]["default"]["url"]
                                else:
                                    c_name = c_name or "Unknown"
                                    c_thumb = c_thumb or ""
                                
                            if any(c.get("id") == c_id for c in st.session_state.channels):
                                st.warning(f"Channel '{c_name}' is already being tracked.")
                            else:
                                st.session_state.channels.append({"name": c_name, "id": c_id, "thumbnail": c_thumb})
                                save_channels(st.session_state.channels)
                                st.success(f"Added {c_name}!")
                                st.rerun()

    st.divider()
    
    # Display existing channels with delete buttons
    if not st.session_state.channels:
        st.info("No channels are currently being tracked.")
    else:
        for i, channel in enumerate(st.session_state.channels):
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                show_channel_image(channel.get("thumbnail"))
            with col2:
                st.markdown(f"**{channel['name']}**  \n`{channel['id']}`")
            with col3:
                if st.button("Remove", key=f"del_{i}", type="secondary", use_container_width=True):
                    st.session_state.channels.pop(i)
                    save_channels(st.session_state.channels)
                    st.rerun()
            st.divider()


# === TAB 3: Discover ===
with tab3:
    st.header("Discover Podcasts")
    st.markdown("Search for new channels to track by topic or keyword.")
    
    discover_query = st.text_input("Enter a topic (e.g. 'AI Startups', 'Nutrition', 'Finance')", key="discover_input")
    
    if st.button("Search YouTube", use_container_width=True):
        if discover_query:
            with st.spinner("Searching YouTube..."):
                try:
                    youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
                    resp = youtube.search().list(part="snippet", type="channel", q=discover_query, maxResults=10).execute()
                    
                    st.session_state.search_results = []
                    items = resp.get("items", [])
                    for item in items:
                        snippet = item["snippet"]
                        st.session_state.search_results.append({
                            "name": snippet["title"],
                            "id": snippet["channelId"],
                            "desc": snippet.get("description", "No description provided."),
                            "thumb": snippet["thumbnails"]["default"]["url"]
                        })
                except Exception as e:
                    st.error(f"Error searching YouTube: {e}")

    # Display Search Results from Session State
    if st.session_state.search_results:
        st.subheader("Search Results")
        for item in st.session_state.search_results:
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                show_channel_image(item["thumb"])
            with col2:
                st.markdown(f"**{item['name']}**")
                st.caption(item['desc'][:150] + ("..." if len(item['desc'])>150 else ""))
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if any(c.get("id") == item["id"] for c in st.session_state.channels):
                    st.button("Tracking", disabled=True, key=f"btn_track_{item['id']}")
                else:
                    if st.button("Add", key=f"btn_add_{item['id']}", type="primary"):
                        st.session_state.channels.append({
                            "name": item["name"], 
                            "id": item["id"], 
                            "thumbnail": item["thumb"]
                        })
                        save_channels(st.session_state.channels)
                        st.success(f"Added {item['name']}!")
                        # Don't rerun immediately to stay in Discover tab and see other results
                        # But we need to update state so "Add" becomes "Tracking"
                        st.rerun()
            st.divider()
                    
    st.divider()
    st.subheader("Recommended for You")
    
    if st.button("Refresh Recommendations") or not st.session_state.recommendations:
        if st.session_state.channels:
            import random
            seed_channel = random.choice(st.session_state.channels)["name"]
            st.session_state.recommend_seed = seed_channel
            
            with st.spinner(f"Fetching recommendations based on {seed_channel}..."):
                try:
                    youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
                    resp = youtube.search().list(
                        part="snippet", 
                        type="channel", 
                        q=f"podcasts like {seed_channel}", 
                        maxResults=5
                    ).execute()
                    
                    st.session_state.recommendations = []
                    for item in resp.get("items", []):
                        snippet = item["snippet"]
                        if snippet["channelId"] not in [c["id"] for c in st.session_state.channels]:
                            st.session_state.recommendations.append({
                                "name": snippet["title"],
                                "id": snippet["channelId"],
                                "desc": snippet.get("description", ""),
                                "thumb": snippet["thumbnails"]["default"]["url"]
                            })
                except Exception as e:
                    st.error(f"Error fetching recommendations: {e}")
        else:
            st.info("Add some channels to get recommendations!")

    if st.session_state.recommendations:
        st.write(f"*Based on your interest in **{st.session_state.get('recommend_seed', '')}***:")
        for item in st.session_state.recommendations:
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                show_channel_image(item["thumb"])
            with col2:
                st.markdown(f"**{item['name']}**")
                st.caption(item['desc'][:150] + ("..." if len(item['desc'])>150 else ""))
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Add", key=f"btn_rec_{item['id']}", type="secondary"):
                    st.session_state.channels.append({"name": item["name"], "id": item["id"]})
                    save_channels(st.session_state.channels)
                    st.success(f"Added {item['name']}!")
                    # Filter out from recommendations list in session state
                    st.session_state.recommendations = [r for r in st.session_state.recommendations if r['id'] != item['id']]
                    st.rerun()
            st.divider()

# === TAB 4: Pipeline & Queue ===
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Pipeline Execution")
        
        # Lookback Period Selector
        lookback_options = {
            "24 Hours (Daily)": 24,
            "48 Hours": 48,
            "7 Days (Weekly)": 168,
            "14 Days": 336,
            "30 Days (Monthly)": 720
        }
        selected_label = st.selectbox("Discovery Lookback Period", options=list(lookback_options.keys()), index=0)
        lookback_hours = lookback_options[selected_label]
        
        st.markdown(f"The pipeline will search for videos published in the last **{selected_label}**.")
        
        if st.button("▶️ Run Pipeline Now", type="primary", use_container_width=True):
            # Create a "Mission Control" UI for the run
            st.markdown("### 🚀 Mission Control")
            p_col1, p_col2, p_col3 = st.columns(3)
            
            with p_col1:
                p1_status = st.empty()
                p1_status.markdown("📡 **Discovery**  \n`Waiting...`")
            with p_col2:
                p2_status = st.empty()
                p2_status.markdown("🏗️ **Extraction**  \n`Waiting...`")
            with p_col3:
                p3_status = st.empty()
                p3_status.markdown("🧠 **Summarization**  \n`Waiting...`")
                
            progress_bar = st.progress(0, text="Initializing...")
            
            # Placeholder for active thumbnail
            thumb_placeholder = st.empty()
            
            log_expander = st.expander("🛠️ Live Process Logs", expanded=True)
            
            try:
                env = os.environ.copy()
                env["DISCOVERY_LOOKBACK_HOURS"] = str(lookback_hours)
                
                process = subprocess.Popen(
                    ["python", "main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env
                )
                
                # Active tracking of phases
                for line in iter(process.stdout.readline, ""):
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                        
                    # Live Log Display
                    display_line = clean_line
                    if " - INFO - " in clean_line:
                        display_line = clean_line.split(" - INFO - ")[-1]
                    elif " - WARNING - " in clean_line:
                        display_line = f"⚠️ {clean_line.split(' - WARNING - ')[-1]}"
                    elif " - ERROR - " in clean_line:
                        display_line = f"❌ {clean_line.split(' - ERROR - ')[-1]}"
                    
                    log_expander.text(display_line)
                    
                    # Thumbnail Logic: If we see an extraction line, try to grab the thumbnail
                    if "Extracting audio for" in display_line or "Summarizing natively via Audio:" in display_line:
                        # Extract the ID or Title between parentheses/after colon
                        import re
                        match = re.search(r"\(([A-Za-z0-9_-]+)\)", display_line) or re.search(r"Audio: (.*)", display_line)
                        if match:
                            target = match.group(1).strip()
                            # Look up in the queue to find the thumb
                            try:
                                queue_data = load_queue()
                                current_v = next((v for v in queue_data if v['id'] == target or v['title'] == target), None)
                                if current_v and current_v.get('thumbnail'):
                                    with thumb_placeholder.container():
                                        st.markdown(f"**Currently Processing:** {current_v['title']}")
                                        st.image(current_v['thumbnail'], width=300)
                                        st.divider()
                            except:
                                pass

                    # Phase Tracker Logic
                    if "Phase 1: Discovery" in clean_line:
                        p1_status.markdown("📡 **Discovery**  \n`Running... ⚙️`")
                        progress_bar.progress(10, text="Scanning YouTube channels...")
                    elif "Discovery complete" in clean_line:
                        p1_status.markdown("📡 **Discovery**  \n`Complete! ✅`")
                        progress_bar.progress(33, text="Videos discovered. Moving to extraction...")
                    
                    elif "Phase 2: Extraction" in clean_line:
                        p2_status.markdown("🏗️ **Extraction**  \n`Working... ⚡`")
                        progress_bar.progress(40, text="Downloading high-speed audio...")
                    elif "Extraction complete" in clean_line:
                        p2_status.markdown("🏗️ **Extraction**  \n`Finished! ✅`")
                        thumb_placeholder.empty() # Clear thumb after extraction phase? Or keep for summary?
                        progress_bar.progress(66, text="Audio ready for AI analysis...")
                        
                    elif "Phase 3: Summarization" in clean_line:
                        p3_status.markdown("🧠 **Summarization**  \n`Thinking... 🕵️`")
                        progress_bar.progress(75, text="Gemini 2.5 is listening to audio...")
                    elif "Summarization complete" in clean_line:
                        p3_status.markdown("🧠 **Summarization**  \n`Brief Generated! ✅`")
                        thumb_placeholder.empty()
                        progress_bar.progress(100, text="All systems clear.")
                
                process.stdout.close()
                return_code = process.wait()
                
                if return_code == 0:
                    st.success("Pipeline executed successfully!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Pipeline run failed. Check the logs above for details.")
                    
            except Exception as e:
                st.error(f"Error running pipeline: {e}")
        
        # Log Viewer Section
        st.divider()
        st.subheader("Pipeline Logs")
        if os.path.exists("data/pipeline.log"):
            with open("data/pipeline.log", "r") as f:
                logs = f.readlines()
                # Show last 100 lines
                st.code("".join(logs[-100:]))
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear Logs", use_container_width=True):
                    with open("data/pipeline.log", "w") as f:
                        pass
                    st.rerun()
            with c2:
                if st.button("🔄 Reset Discovery History", help="Clears the database of already processed videos, allowing them to be found again.", use_container_width=True):
                    db_path = os.path.join("data", "processed_videos.json")
                    if os.path.exists(db_path):
                        os.remove(db_path)
                    st.success("History cleared! Next run will find all recent videos again.")
                    st.rerun()
        else:
            st.info("No logs available yet. Run the pipeline to generate logs.")

    with col2:
        if queue:
            st.header("⏳ Pending Briefing")
            st.warning(f"Total: {len(queue)} videos found but not yet briefed.")
            st.markdown("These videos have been discovered but the final brief was not generated. You can run the pipeline to process them.")
            
            if st.button("🗑️ Clear All Pending Items", use_container_width=True):
                with open("data/queue.json", "w") as f:
                    json.dump([], f)
                st.rerun()
                
            for i, video in enumerate(queue):
                with st.expander(f"📺 {video.get('title', 'Unknown')}", expanded=(i==0)):
                    st.markdown(f"**Channel:** {video.get('channel', 'Unknown')}")
                    
                    if video.get('duration_minutes'):
                        st.markdown(f"**Duration:** {video.get('duration_minutes'):.1f} mins")
                        
                    st.markdown(f"[Watch on YouTube]({video.get('url', '#')})")
                    
                    # Status indicators based on keys
                    status = "Pending Extraction"
                    if video.get('audio_path'):
                        status = "Extracted (Waiting for AI Briefing)"
                        
                    st.markdown(f"**Status:** `{status}`")
        else:
            st.header("✅ System Ready")
            st.success("All discovered videos have been successfully briefed. There are no items pending.")
            st.caption("New videos will appear here temporarily during the next discovery cycle.")
