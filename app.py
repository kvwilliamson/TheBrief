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

DEFAULT_CATEGORIES = [
    "General Financial Investing and Speculation",
    "Precious Metals",
    "Artificial Intelligence",
    "Health and Nutrition",
    "Philosophy and Thoughtfulness",
    "Other"
]

def get_all_categories():
    """Returns dynamic list of categories: Defaults + any custom ones found in channels."""
    cats = DEFAULT_CATEGORIES.copy()
    current_channels = st.session_state.get('channels', [])
    for c in current_channels:
        cat = c.get("category")
        if cat and cat not in cats:
            # Insert before "Other" if possible
            if "Other" in cats:
                cats.insert(cats.index("Other"), cat)
            else:
                cats.append(cat)
    return cats

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
    /* 
       ROBUST HEADER STYLING 
       Intelligence Blue Categories
    */
    .st-emotion-cache-p3m996, .st-emotion-cache-1pxm6i, [data-testid="stExpander"] summary p {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        color: #1c83e1 !important; /* Intelligence Blue */
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 
       CONTROLLED DENSITY 
       Ensures thumbnails have breathing room (Tightened by half)
    */
    [data-testid="stExpander"] [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] {
        gap: 6px !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }
    
    [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
        gap: 6px !important;
    }

    /* Standard weight for buttons */
    .stButton button div p {
        font-weight: 400 !important;
    }
    
    hr {
        margin-top: 7px !important;
        margin-bottom: 7px !important;
        border: 0;
        border-top: 1px solid #444;
    }
    </style>
""", unsafe_allow_html=True)


# --- UI Layout ---

st.title("🎙️ TheBrief Dashboard")
st.markdown("Your daily deep-dive podcast briefing system.")

tab1, tab2, tab3 = st.tabs(["📑 Daily Briefs", "📺 Sources", "⚙️ Pipeline & Queue"])


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


# === TAB 2: Sources ===
with tab2:
    st.header("📡 Intelligence Sources")
    channels = load_channels()
    
    # Discovery & New Sources Expander
    with st.expander("🔍 Discovery & New Sources", expanded=False):
        st.subheader("Discover New Sources")
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
                        # Show category selector before adding
                        with st.popover("Add", use_container_width=True):
                            cats = get_all_categories()
                            selected_cat = st.selectbox("Category", options=cats + ["[+] Create New..."], index=cats.index("Other") if "Other" in cats else 0, key=f"cat_{item['id']}")
                            
                            final_cat = selected_cat
                            if selected_cat == "[+] Create New...":
                                final_cat = st.text_input("New Category Name", key=f"new_cat_{item['id']}")
                                
                            if st.button("Confirm Add", key=f"conf_{item['id']}", use_container_width=True, type="primary"):
                                if selected_cat == "[+] Create New..." and not final_cat.strip():
                                    st.error("Please enter a category name.")
                                else:
                                    st.session_state.channels.append({
                                        "name": item["name"], 
                                        "id": item["id"], 
                                        "thumbnail": item["thumb"],
                                        "category": final_cat.strip()
                                    })
                                    save_channels(st.session_state.channels)
                                    st.success(f"Added {item['name']} to {final_cat}!")
                                    st.rerun()
                st.divider()
                        
        st.divider()
        st.subheader("💡 Recommended for You")
        
        if st.button("Refresh Recommendations") or not st.session_state.recommendations:
            if st.session_state.channels:
                import random
                seed_channel = random.choice(st.session_state.channels)["name"]
                st.session_state.recommend_seed = seed_channel
                
                with st.spinner(f"Fetching recommendations based on {seed_channel}..."):
                    try:
                        youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
                        # Handle case where developerKey might be missing in env but checked later
                        if not os.getenv("YOUTUBE_API_KEY"):
                            st.error("YOUTUBE_API_KEY missing in .env")
                        else:
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
                    with st.popover("Add", use_container_width=True):
                        cats = get_all_categories()
                        rec_cat = st.selectbox("Category", options=cats + ["[+] Create New..."], index=cats.index("Other") if "Other" in cats else 0, key=f"rec_cat_{item['id']}")
                        
                        final_rec_cat = rec_cat
                        if rec_cat == "[+] Create New...":
                            final_rec_cat = st.text_input("New Category Name", key=f"rec_new_cat_{item['id']}")
                            
                        if st.button("Confirm Add", key=f"rec_conf_{item['id']}", use_container_width=True, type="primary"):
                            if rec_cat == "[+] Create New..." and not final_rec_cat.strip():
                                st.error("Please enter a category name.")
                            else:
                                st.session_state.channels.append({
                                    "name": item["name"], 
                                    "id": item["id"],
                                    "thumbnail": item["thumb"],
                                    "category": final_rec_cat.strip()
                                })
                                save_channels(st.session_state.channels)
                                st.success(f"Added {item['name']} to {final_rec_cat}!")
                                # Filter out from recommendations list in session state
                                st.session_state.recommendations = [r for r in st.session_state.recommendations if r['id'] != item['id']]
                                st.rerun()
                st.divider()

    st.divider()
    
    # Display existing channels with delete buttons
    if not st.session_state.channels:
        st.info("No channels are currently being tracked.")
    else:
        # Group channels by category
        from collections import defaultdict
        grouped_channels = defaultdict(list)
        for channel in st.session_state.channels:
            cat = channel.get("category", "Other")
            grouped_channels[cat].append(channel)
        
        # Display each category
        all_cats = get_all_categories()
        for category in all_cats:
            cat_channels = grouped_channels.get(category, [])
            if cat_channels:
                with st.expander(f"📁 {category} ({len(cat_channels)})", expanded=True):
                    for i, channel in enumerate(cat_channels):
                        # Find original index in st.session_state.channels for deletion
                        orig_index = next((idx for idx, c in enumerate(st.session_state.channels) if c['id'] == channel['id']), None)
                        
                        col1, col2, col3 = st.columns([0.5, 8.5, 1])
                        with col1:
                            show_channel_image(channel.get("thumbnail"))
                        with col2:
                            st.markdown(f"**{channel['name']}**")
                        with col3:
                            with st.popover("Category", use_container_width=True):
                                cats = get_all_categories()
                                current_cat = channel.get("category", "Other")
                                cat_idx = cats.index(current_cat) if current_cat in cats else cats.index("Other")
                                
                                new_cat_choice = st.selectbox(
                                    "Move to Sector", 
                                    options=cats + ["[+] Create New..."], 
                                    index=cat_idx,
                                    key=f"move_{channel['id']}"
                                )
                                
                                final_move_cat = new_cat_choice
                                if new_cat_choice == "[+] Create New...":
                                    final_move_cat = st.text_input("New Sector Name", key=f"move_new_input_{channel['id']}")
                                
                                # Move action
                                if new_cat_choice == "[+] Create New...":
                                    if st.button("Confirm New Sector", key=f"move_conf_{channel['id']}", use_container_width=True):
                                        if final_move_cat.strip():
                                            if orig_index is not None:
                                                st.session_state.channels[orig_index]["category"] = final_move_cat.strip()
                                                save_channels(st.session_state.channels)
                                                st.rerun()
                                        else:
                                            st.error("Enter name")
                                elif new_cat_choice != current_cat:
                                    if orig_index is not None:
                                        st.session_state.channels[orig_index]["category"] = new_cat_choice
                                        save_channels(st.session_state.channels)
                                        st.toast(f"✅ {channel['name']} moved to {new_cat_choice}")
                                        st.rerun()
                                    
                            if st.button("Remove", key=f"del_{channel['id']}_{i}", type="secondary", use_container_width=True):
                                if orig_index is not None:
                                    st.session_state.channels.pop(orig_index)
                                    save_channels(st.session_state.channels)
                                    st.rerun()
                        # Removing st.divider() for tighter padding
                        st.markdown("---") # Thinner than divider
        
        # Also check for any categories NOT in DEFAULT_CATEGORIES



# === TAB 3: Pipeline & Queue ===
with tab3:
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
                
                total_videos = 0
                extracting_count = 0
                summarizing_count = 0

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
                    
                    # Thumbnail Logic
                    if "Extracting audio for" in display_line or "Generating Brief for" in display_line:
                        import re
                        match_id = re.search(r"\(([A-Za-z0-9_-]+)\)", display_line)
                        if match_id:
                            target_id = match_id.group(1)
                            try:
                                queue_data = load_queue()
                                current_v = next((v for v in queue_data if v['id'] == target_id), None)
                                if current_v and current_v.get('thumbnail'):
                                    with thumb_placeholder.container():
                                        title_prefix = "AI listening to: " if "Generating" in display_line else "Extracting: "
                                        st.markdown(f"**{title_prefix}** {current_v['title']}")
                                        st.image(current_v['thumbnail'], width=400)
                                        st.divider()
                            except: pass

                    import re
                    
                    # Discovery Count Capture
                    if "Discovery complete. Added" in display_line:
                        match_total = re.search(r"Added (\d+) new videos", display_line)
                        if match_total:
                            total_videos = int(match_total.group(1))
                            p1_status.markdown(f"📡 **Discovery**  \n`Found {total_videos} assets ✅`")
                            progress_bar.progress(33, text=f"Search complete. {total_videos} videos queued.")

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
                    elif "Extracting audio for" in display_line:
                        extracting_count += 1
                        msg = f"Extracting {extracting_count} of {total_videos}" if total_videos > 0 else f"Extracting {extracting_count}..."
                        p2_status.markdown(f"🏗️ **Extraction**  \n`{msg} ⚡`")
                        p_val = 33 + int((extracting_count / (total_videos if total_videos > 0 else 10)) * 33)
                        progress_bar.progress(min(p_val, 66), text=f"🏗️ {msg}: {display_line.split('for ')[-1]}")
                    elif "Extraction complete" in clean_line:
                        match_time = re.search(r"\(Time: ([\d.]+)s\)", clean_line)
                        timer_str = f" ({match_time.group(1)}s)" if match_time else ""
                        p2_status.markdown(f"🏗️ **Extraction**  \n`Finished!{timer_str} ✅`")
                        thumb_placeholder.empty()
                        progress_bar.progress(66, text="Audio ready for AI analysis...")
                        
                    elif "Phase 3: Summarization" in clean_line:
                        p3_status.markdown("🧠 **Summarization**  \n`Thinking... 🕵️`")
                        progress_bar.progress(70, text="Gemini 2.5 is listening to audio...")
                    elif "Generating Brief for" in display_line:
                        summarizing_count += 1
                        msg = f"Auditing {summarizing_count} of {total_videos}" if total_videos > 0 else f"Auditing {summarizing_count}..."
                        p3_status.markdown(f"🧠 **Summarization**  \n`{msg} 🕵️`")
                        p_val = 66 + int((summarizing_count / (total_videos if total_videos > 0 else 10)) * 34)
                        progress_bar.progress(min(p_val, 100), text=f"🧠 {msg}: {display_line.split('for ')[-1]}")
                    elif "Summarization complete" in clean_line:
                        match_time = re.search(r"\(Time: ([\d.]+)s\)", clean_line)
                        timer_str = f" ({match_time.group(1)}s)" if match_time else ""
                        p3_status.markdown(f"🧠 **Summarization**  \n`Brief Generated!{timer_str} ✅`")
                        thumb_placeholder.empty()
                        progress_bar.progress(100, text="All systems clear.")
                
                process.stdout.close()
                return_code = process.wait()
                
                if return_code == 0:
                    st.success("Pipeline executed successfully!")
                    st.balloons()
                    if st.button("✨ All Done! Refresh Dashboard", type="primary"):
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
