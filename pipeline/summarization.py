import os
import json
import logging
import concurrent.futures
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from google import genai
import time
from pipeline.clustering import perform_semantic_clustering
from pipeline.clustering import logger as cluster_logger
# Removed pipeline.profiles import as part of hardcode removal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def get_next_percentile():
    """Calculates optimal percentile based on last run's fragmentation metrics."""
    stats_path = "data/clustering_stats.json"
    if not os.path.exists(stats_path):
        return 85 
    
    try:
        with open(stats_path, "r") as f:
            history = json.load(f)
        if not history:
            return 85
            
        last_run = history[-1]
        percentile = last_run.get("percentile", 85)
        video_count = last_run.get("video_count", 1)
        if video_count < 5: return percentile # Too small to tune
        
        singleton_ratio = last_run.get("singleton_count", 0) / video_count
        max_cluster_ratio = last_run.get("max_cluster_size", 0) / video_count
        
        # Auto-tuning guardrails
        if singleton_ratio > 0.6:
            new_p = max(50, percentile - 5)
            logger.info(f"🔄 High fragmentation ({singleton_ratio:.1%}). Tuning percentile: {percentile} -> {new_p}")
            return new_p
        elif max_cluster_ratio > 0.5:
            new_p = min(95, percentile + 5)
            logger.info(f"🔄 Over-merging ({max_cluster_ratio:.1%}). Tuning percentile: {percentile} -> {new_p}")
            return new_p
            
        return percentile
    except Exception as e:
        logger.warning(f"Percentile tuning failed: {e}")
        return 85

# --- Architectural Constants ---
SUMMARY_MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "text-embedding-004"
GENAI_CLIENT = None

def get_genai_client():
    global GENAI_CLIENT
    if GENAI_CLIENT is None:
        GENAI_CLIENT = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))
    return GENAI_CLIENT

ALLOWED_BIAS = {"constructive", "defensive", "neutral", "counter-consensus"}

# --- High-Utility Decision Format (vNext) Schemas ---

class BriefSchema(BaseModel):
    episode_title: str = Field(description="Title of the episode")
    channel: str = Field(description="Name of the channel")
    duration_minutes: int = Field(description="Duration in minutes")
    podcast_date: str = Field(description="Date episode was published (from context)")
    processing_date: str = Field(description="Date TheBrief processed it (today's date)")
    one_line_summary: str = Field(description="A single sentence summarizes the core thesis.")
    core_claims: List[str] = Field(description="Significant specific claims made in the audio. Typically 2-6.")
    signal_strength: int = Field(ge=1, le=10, description="Intelligence signal strength score 1-10.")
    themes: List[str] = Field(description="Short tags representing the clusterable themes. Typically 2-6.")
    positioning_implication: str = Field(description="What is the suggested action or bias shift? Max 2 sentences.")
    time_horizon: str = Field(description="Strictly one of: short, medium, long")
    shelf_life: str = Field(description="Strictly: Short, Medium, Long")
def get_llm():
    model_choice = os.getenv("SUMMARY_MODEL", "gemini").lower()
    
    if model_choice == "gemini":
        try:
            # We must use flash for native audio understanding
            return ChatGoogleGenerativeAI(
                model=SUMMARY_MODEL_NAME,
                temperature=0.2,
                google_api_key=os.getenv("GOOGLE_AI_API_KEY")
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}. Falling back to OpenAI.")
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0.2,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
    else:
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

def normalize_channel_name(name: str) -> str:
    """Removes markdown bold markers and strips whitespace for robust matching."""
    if not name:
        return ""
    # Standardize common channel quirks (like bold/white)
    clean = name.replace("**", "").strip()
    return clean

def summarize_transcript(video, llm):
    # Start briefing
    parser = JsonOutputParser(pydantic_object=BriefSchema)
    
    # 1. Upload Audio to Gemini
    audio_path = video.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Missing audio path for {video['title']}")
        return None
        
    client = get_genai_client()
    file_handle = client.files.upload(file=audio_path)
    
    # Wait for processing
    while file_handle.state == "PROCESSING":
        time.sleep(2)
        file_handle = client.files.get(name=file_handle.name)
        
    if file_handle.state == "FAILED":
        logger.error(f"Gemini processing failed for {video['title']}")
        return None
        
    logger.info(f"Generating Brief for {video['title']} ({video['id']})...")
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    template = (
        "You are an expert intelligence analyst.\n"
        "Generate a high-utility, educational, and decision-oriented Intelligence Brief directly from the attached audio.\n\n"
        "### 🛡️ INDEPENDENT TOPIC DOMINANCE RULE:\n"
        "This is an ISOLATED analysis. Your EXCLUSIVE source of truth is the ATTACHED AUDIO.\n"
        "1. IGNORE the general theme of this channel or other videos.\n"
        "2. The current Title {title} and Channel {channel} are for reference only. THE AUDIO WINS ANY CONTRADICTIONS.\n\n"
        "### BRIEFING REQUIREMENTS:\n"
        "- ONE LINE SUMMARY: A single sentence distilling the core signal.\n"
        "- CORE CLAIMS: Specific data points, projections, or claims made in the audio. Avoid filler.\n"
        "- SIGNAL STRENGTH: Score 1-10 based on uniqueness and actionability of information.\n"
        "- THEMES: lowercase short tags. NO PREDEFINED LIST. Create the taxonomy based on the content.\n"
        "- POSITIONING IMPLICATION: Direct consequence of this intelligence on a listener's strategy. Be domain-agnostic (e.g., if content is about health, biased towards a protocol; if financial, biased towards a position).\n"
        "- TIME HORIZON: short | medium | long.\n\n"
        "STRICT CONSTRAINTS:\n"
        "1. Tone: Professional intelligence memo. No hype, no emojis.\n"
        "2. Max word count: 400-600 words.\n"
        "3. Output MUST conform exactly to the JSON schema.\n\n"
        "Video metadata for identification:\nTitle: {title}\nChannel: {channel}\nToday: {today}\n\n"
        "Format instructions:\n{format_instructions}\n"
    )
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["title", "channel", "duration_minutes", "today"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    try:
        # 0. Format the prompt string
        formatted_prompt = prompt.format(
            title=video.get("title", ""),
            channel=video.get("channel", ""),
            duration_minutes=video.get("duration_minutes", 0),
            today=today_str
        )
        
        # 1. Build Native Google GenAI Request
        # The official SDK is more robust for audio than the LangChain wrapper
        response = client.models.generate_content(
            model=SUMMARY_MODEL_NAME,
            contents=[
                formatted_prompt,
                file_handle
            ],
            config={
                "response_mime_type": "application/json",
            }
        )
        
        # Parse the JSON response
        try:
            import json
            result = json.loads(response.text)
            return result
        except OutputParserException as e:
            logger.error(f"Schema parsing error for {video['title']}: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error summarising audio {video['title']}: {e}")
        return None
    finally:
        # 3. Clean up the file from the File API
        try:
            client.files.delete(name=file_handle.name)
        except Exception as e:
            logger.warning(f"Failed to delete file from Gemini API: {e}")
            
        # Delete local copy
        try:
            os.remove(audio_path)
        except OSError:
            pass

def get_embedding(text: str) -> List[float]:
    """Generates an embedding for the given text using Gemini's embedding model."""
    if not text:
        return []
    try:
        client = get_genai_client()
        # Using text-embedding-004 as it is modern and efficient
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=text
        )
        return response.embeddings[0].values
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

def format_markdown(brief: dict, video_url: str = "", thumbnail_url: str = "") -> str:
    title = brief.get('episode_title', 'Unknown Title')
    
    md = ""
    if video_url:
        md += f"## [{title}]({video_url})\n"
    else:
        md += f"## {title}\n"
        
    if thumbnail_url:
        md += f"![{title}]({thumbnail_url})\n\n"

    md += f"**{brief.get('channel', 'Unknown')}** | **Length:** {brief.get('duration_minutes', 0)} min\n"
    md += f"**Published:** {brief.get('podcast_date', 'N/A')} | **Processed:** {brief.get('processing_date', 'N/A')} | **Shelf Life:** {brief.get('shelf_life', 'N/A')}\n\n"
    
    md += "### Quick-Scan Metrics\n"
    md += f"- **Signal Strength:** {brief.get('signal_strength', 0)}/10\n"
    md += f"- **Time Horizon:** {brief.get('time_horizon', 'N/A')}\n\n"

    md += f"**Thesis:** {brief.get('one_line_summary')}\n\n"

    themes = brief.get('themes', [])
    if themes:
        md += f"**Themes:** {', '.join([f'`#{t}`' for t in themes])}\n\n"

    claims = brief.get('core_claims', [])
    if claims:
        md += "### Core Intelligence Claims\n"
        for claim in claims:
            md += f"- {claim}\n"
        md += "\n"

    md += "### Positioning Implication\n"
    md += f"{brief.get('positioning_implication')}\n\n"

    md += "---\n\n"
    return md

def format_html(brief: dict, video_url: str = "", thumbnail_url: str = "") -> str:
    title = brief.get('episode_title', 'Unknown Title')
    
    # High-Contrast White Background Container
    html = f"<div style='border-bottom: 1px solid #ddd; padding: 15px 0; background-color: #ffffff; color: #202124; font-family: Arial, Helvetica, sans-serif; line-height: 1.5;'>"
    
    # Title (Link Blue)
    title_html = f"<a href='{video_url}' style='text-decoration: none; color: #1155CC; font-size: 16px; font-weight: bold;'>{title}</a>" if video_url else f"<span style='font-size: 16px; font-weight: bold;'>{title}</span>"
    html += f"<div style='margin-bottom: 5px;'>{title_html}</div>"
    
    # Header with Thumbnail (If available)
    html += "<table width='100%' style='border-collapse: collapse;'><tr>"
    if thumbnail_url:
        html += f"<td width='130' style='vertical-align: top; padding-right: 15px;'><img src='{thumbnail_url}' width='120' style='border-radius: 4px; border: 1px solid #eee;' /></td>"
    
    html += "<td style='vertical-align: top;'>"
    html += f"<p style='margin: 0 0 5px 0; color: #70757A; font-size: 13px;'><strong>{brief.get('channel', 'Unknown')}</strong> | {brief.get('duration_minutes', 0)} min</p>"
    
    # Metrics Table
    html += "<table width='100%' style='border-top:1px solid #eee; border-bottom:1px solid #eee; margin:10px 0; border-collapse:collapse;'>"
    html += "<tr>"
    html += f"<td style='padding: 8px; text-align: center; font-size: 12px;'><span style='color:#70757A;'>Signal:</span> <b style='color:#1155CC;'>{brief.get('signal_strength')}/10</b></td>"
    html += f"<td style='padding: 8px; text-align: center; font-size: 12px;'><span style='color:#70757A;'>Horizon:</span> <b style='color:#202124;'>{brief.get('time_horizon')}</b></td>"
    html += f"<td style='padding: 8px; text-align: center; font-size: 12px;'><span style='color:#70757A;'>Shelf Life:</span> <b style='color:#F9AB00;'>{brief.get('shelf_life')}</b></td>"
    html += "</tr></table>"

    html += f"<div style='background: #f8f9fa; padding: 10px; border-left: 4px solid #1A73E8; margin: 10px 0; color: #3C4043; font-size: 14px;'><b>Thesis:</b> {brief.get('one_line_summary')}</div>"
    
    themes = brief.get('themes', [])
    if themes:
        themes_str = " ".join([f"<span style='background:#e8f0fe; color:#1a73e8; padding:2px 6px; border-radius:10px; font-size:11px; margin-right:5px;'>#{t}</span>" for t in themes])
        html += f"<div style='margin-top:5px;'>{themes_str}</div>"
    
    html += "</td></tr></table>"

    # Core Claims
    claims = brief.get('core_claims', [])
    if claims:
        html += f"<div style='color:#1A73E8; font-weight:bold; margin-top:15px; font-size: 13px; text-transform:uppercase;'>Core Intelligence Claims</div>"
        for claim in claims:
            html += f"<div style='margin-bottom:8px; font-size: 14px; color: #3C4043;'>• {claim}</div>"

    html += f"<div style='color:#EA4335; font-weight:bold; margin-top:15px; font-size: 13px; text-transform:uppercase;'>Positioning Implication</div>"
    html += f"<div style='margin: 5px 0; font-size: 14px; color: #3C4043;'>{brief.get('positioning_implication')}</div>"

    html += "</div>"
    return html

def generate_meta_summary(clusters_data: List[Dict[str, Any]], total_assets: int, llm) -> str:
    """
    World-Class Intelligence Synthesis Pass.
    Produces high-density, quantified strategic analysis.
    """
    cluster_texts = []
    for c in clusters_data:
        # Dominance quantification
        weight = (c['size'] / total_assets) * 100
        summaries = "\n".join([f"- {b['one_line_summary']} (Signal: {b['signal_strength']})" for b in c.get('briefs', [])])
        
        # Claim extraction for watchlist triggers
        claims = "\n".join([f"  * {claim}" for b in c.get('briefs', []) for claim in b.get('core_claims', [])])
        
        cluster_texts.append(
            f"NARRATIVE: {c['name']} ({c['size']} assets, {weight:.0f}% dominance)\n"
            f"DIAGNOSTICS: Coherence: {c['coherence']:.2f}, Crowding: {c['crowding_label']}, Strength: {c['strength']:.1f}\n"
            f"BIAS: {c['bias']} | CHANNELS: {', '.join(c['channels'])}\n"
            f"SUMMARIES:\n{summaries}\n"
            f"EXTRACTED CLAIMS:\n{claims}"
        )
    
    clusters_input = "\n\n".join(cluster_texts)
    
    system_prompt = (
        "You are a Lead Intelligence Strategist for a Tier-1 institutional desk. "
        "Your task is to produce a DENSE, QUANTIFIED Narrative Radar Report.\n\n"
        "### STYLE GUIDE:\n"
        "- NO qualitative filler (e.g., 'The dominant narrative is...').\n"
        "- NO adverbs or generic macro commentary.\n"
        "- MAX signal density. MIN words.\n"
        "- USE numerical backing for every assertion."
    )
    
    user_content = (
        "### DATASET DIAGNOSTICS:\n"
        f"Total Assets Analyzed: {total_assets}\n\n"
        "### INPUT NARRATIVE CLUSTERS:\n"
        f"{clusters_input}\n\n"
        "### REQUIRED SECTIONS:\n"
        "1. QUANTIFIED DOMINANCE: Identify the top cluster by % dominance. State its Strength and Convergence.\n"
        "2. CROWDING & RISK SATURATION: Identify clusters with HIGH/MODERATE crowding. Assess if sentiment is reaching an 'echo chamber' state.\n"
        "3. INTRA-CLUSTER FRACTURES: Identify clusters where Coherence is < 0.6. Identify specific opposing directional biases or claims.\n"
        "4. CROSS-CLUSTER INTERACTION: Explicitly map how one narrative (e.g., Geopolitics) is propagating into another (e.g., Energy/Metals).\n"
        "5. DATASET-DERIVED WATCHLIST: 2-3 specific triggers (Price targets, deadlines, catalytic events) extracted ONLY from the provided CLAIMS. No generic templates.\n\n"
        "STRICT CONSTRAINTS:\n"
        "- Tone: Clinical, precise, institutional.\n"
        "- Target Reduction: Be 20% shorter than a standard summary.\n"
    )
    
    try:
        from langchain_core.messages import SystemMessage
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ])
        if hasattr(response, 'content'):
            return response.content.strip()
        return str(response)
    except Exception as e:
        logger.error(f"Error generating meta-summary: {e}")
        return "Meta-summary generation failed due to synthesis error."

def generate_cluster_label(cluster_briefs: List[Dict[str, Any]], llm) -> Dict[str, str]:
    """
    Generates a dynamic name, description, and positioning bias for a cluster.
    """
    context = "\n".join([f"- {b['one_line_summary']} (Themes: {', '.join(b['themes'])})" for b in cluster_briefs[:3]])
    
    prompt = (
        "You are a narrative analyst.\n"
        "Look at these top summaries from a semantic cluster and generate a human-readable label.\n\n"
        f"### CLUSTER CONTENT:\n{context}\n\n"
        "### OUTPUT FORMAT (JSON):\n"
        "{{\n"
        "  \"cluster_name\": \"Short descriptive phrase (e.g., 'Fed Pivot Speculation' or 'Renewable Infrastructure Growth')\",\n"
        "  \"description\": \"One sentence summary of why these items are grouped.\",\n"
        "  \"positioning_bias\": \"One of: constructive | defensive | neutral | counter-consensus\"\n"
        "}}\n\n"
        "RULES:\n"
        "1. DO NOT use theme counting.\n"
        "2. Be domain-agnostic. No assumption of asset classes unless stated in input.\n"
        "3. High-density professional tone.\n"
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        import json
        # Extract JSON from potential markdown blocks
        clean_text = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        # Validation: Enforce ALLOWED_BIAS
        if data.get("positioning_bias", "").lower() not in ALLOWED_BIAS:
            data["positioning_bias"] = "neutral"
            
        return data
    except Exception as e:
        logger.error(f"Error generating cluster label: {e}")
        return {
            "cluster_name": "Unnamed Cluster",
            "description": "Dynamic grouping of related narratives.",
            "positioning_bias": "neutral"
        }

def send_email_digest(html_content, date_str):
    # Prepare email
    email_to = os.getenv("EMAIL_TO")
    email_from = os.getenv("EMAIL_FROM")
    smtp_host = os.getenv("SMTP_HOST")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    if not all([email_to, email_from, smtp_host, smtp_password]):
        logger.info("Missing email configuration, skipping email delivery.")
        return
        
    try:
        host = smtp_host
        port = 587
        if ":" in smtp_host:
            host, port_str = smtp_host.split(":")
            port = int(port_str)
            
        server = smtplib.SMTP(host, port)
        server.starttls()
        smtp_user = os.getenv("SMTP_USER", email_from)
        server.login(smtp_user, smtp_password)
        
        recipients = [email.strip() for email in email_to.split(",")]
        for recipient in recipients:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"TheBrief Daily Digest - {date_str}"
            msg["From"] = email_from
            msg["To"] = recipient
            msg.attach(MIMEText(html_content, "html"))
            server.sendmail(email_from, recipient, msg.as_string())
            logger.info(f"Email digest sent successfully to {recipient}.")
            
        server.quit()
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def run_summarization():
    queue_path = os.path.join("data", "queue.json")
    if not os.path.exists(queue_path):
        logger.error("No queue found. Run extraction first.")
        return []
        
    with open(queue_path, "r") as f:
        queue = json.load(f)
        
    if not queue:
        logger.info("Queue is empty. No transcripts to summarize.")
        return []
        
    llm = get_llm()
    processed_queue = []
    formatted_briefs = []
    start_time = time.time()

    def process_video_summary(video):
        brief = summarize_transcript(video, llm)
        if brief:
            video["brief"] = brief
            md = format_markdown(brief, video.get('url', ''), video.get('thumbnail', ''))
            html_comp = format_html(brief, video.get('url', ''), video.get('thumbnail', ''))
            
            # Generate embedding for future clustering
            text_for_embedding = f"{brief.get('one_line_summary', '')} {' '.join(brief.get('core_claims', []))}"
            video["embedding"] = get_embedding(text_for_embedding)
            
            logger.info(f"✅ Brief successfully generated for {video['title']}")
            return video, (md, html_comp)
        return None, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_video = {executor.submit(process_video_summary, video): video for video in queue}
        for future in concurrent.futures.as_completed(future_to_video):
            v_res, format_res = future.result()
            if v_res and format_res:
                processed_queue.append(v_res)
                formatted_briefs.append((v_res, format_res))
                
    if not processed_queue:
        logger.warning("No briefs were successfully generated.")
        return []

    # --- PHASE 2: SEMANTIC CLUSTERING (v2: Self-Tuning) ---
    current_percentile = get_next_percentile()
    linkage_mode = os.getenv("CLUSTERING_LINKAGE", "complete")
    processed_queue = perform_semantic_clustering(
        processed_queue, 
        percentile=current_percentile,
        linkage=linkage_mode
    )

    # --- PHASE 3: META SYNTHESIS ---
    # A. Calculate Relative Signal
    max_signal = max([v['brief']['signal_strength'] for v in processed_queue]) if processed_queue else 1
    for v in processed_queue:
        v['brief']['relative_signal'] = v['brief']['signal_strength'] / max_signal

    # B. Narrative Convergence & Crowding Detection
    from collections import defaultdict
    clusters = defaultdict(list)
    for v in processed_queue:
        if v.get('cluster_id') != -1:
            clusters[v['cluster_id']].append(v)
    
    cluster_metrics = []
    cluster_sizes = [len(briefs) for briefs in clusters.values()]
    percentile = int(os.getenv("CROWDING_PERCENTILE", 75))
    crowding_threshold = np.percentile(cluster_sizes, percentile) if cluster_sizes else 0
    
    total_assets = len(processed_queue)
    for c_id, briefs in clusters.items():
        avg_signal = np.mean([b['brief']['signal_strength'] for b in briefs])
        avg_coherence = np.mean([b.get('cluster_coherence', 1.0) for b in briefs])
        cluster_strength = np.mean([b.get('cluster_strength', 0) for b in briefs])
        
        # Crowding Index = (cluster size * avg intra-cluster similarity * avg signal score)
        crowding_index = len(briefs) * avg_coherence * avg_signal
        
        # Quantify Crowding Label
        if crowding_index > 40: crowding_label = "HIGH"
        elif crowding_index > 15: crowding_label = "MODERATE"
        else: crowding_label = "LOW"

        # Dynamic Cluster Labeling
        label_data = generate_cluster_label(briefs, llm)
        
        cluster_metrics.append({
            "id": c_id,
            "name": label_data.get("cluster_name"),
            "description": label_data.get("description"),
            "bias": label_data.get("positioning_bias"),
            "size": len(briefs),
            "dominance_pct": (len(briefs) / total_assets) * 100 if total_assets > 0 else 0,
            "strength": float(cluster_strength),
            "coherence": float(avg_coherence),
            "crowding_index": float(crowding_index),
            "crowding_label": crowding_label,
            "channels": list(set([b['brief']['channel'] for b in briefs])),
            "avg_signal": float(avg_signal),
            "is_crowded": crowding_label == "HIGH",
            "is_singleton": len(briefs) == 1,
            "is_emergent": len(briefs) >= 2 and avg_coherence > 0.85,
            "briefs": [b['brief'] for b in briefs]
        })
    
    # Sort cluster metrics by size (Dominance)
    cluster_metrics = sorted(cluster_metrics, key=lambda x: x['size'], reverse=True)
    
    # C. Executive Meta Summary (Institutional Strategic Pass)
    meta_summary = generate_meta_summary(cluster_metrics, total_assets, llm)

    # Update processed_videos DB
    db_path = os.path.join("data", "processed_videos.json")
    try:
        from tinydb import TinyDB
        db = TinyDB(db_path)
        for v in processed_queue:
            db.insert({"id": v["id"], "title": v["title"], "processed_at": datetime.now().isoformat()})
    except Exception as e:
        logger.warning(f"Note: Error updating processed_videos db: {e}")

    # Compile final outputs
    date_str = datetime.now().strftime("%Y-%m-%d")
    md_filename = os.path.join("briefs", f"{date_str}.md")
    os.makedirs("briefs", exist_ok=True)
    
    total_videos = len(processed_queue)
    total_time = sum(v.get('duration_minutes', 0) for v in processed_queue)
    
    # Header
    summary_md = f"# TheBrief Daily Dispatch - {date_str}\n\n"
    summary_md += f"### 📊 At a Glance\n"
    summary_md += f"- **Total Intelligence Assets:** {total_videos} videos\n"
    summary_md += f"- **Total Subject Time:** {total_time:.1f} minutes\n\n"
    
    # HTML Header
    summary_html = f"<html><body style='font-family: Arial, Helvetica, sans-serif; color: #202124; background-color: #ffffff; padding: 20px;'>"
    summary_html += f"<h1 style='color: #202124; font-size: 24px; margin-bottom: 5px;'>TheBrief Daily Dispatch</h1>"
    summary_html += f"<div style='color: #70757A; margin-bottom: 25px; font-size: 14px;'>Intelligence Report: {date_str}</div>"
    
    summary_html += f"<div style='background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 30px;'>"
    summary_html += f"<h3 style='margin-top: 0; color: #202124; font-size: 18px;'>📊 At a Glance</h3>"
    summary_html += f"<ul style='margin-bottom: 0; font-size: 14px;'>"
    summary_html += f"<li><strong>Total Intelligence Assets:</strong> {total_videos} videos</li>"
    summary_html += f"<li><strong>Total Subject Time:</strong> {total_time:.1f} minutes</li></ul></div>"

    # --- EXECUTIVE META SUMMARY SECTION ---
    summary_md += f"## 🧠 Executive Meta Summary\n\n{meta_summary}\n\n"
    summary_html += f"<div style='background: #1A73E8; color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px;'>"
    summary_html += f"<h2 style='margin-top: 0; font-size: 20px;'>🧠 Executive Meta Summary</h2>"
    summary_html += f"<div style='font-size: 14px; line-height: 1.6;'>{meta_summary.replace(chr(10), '<br/>')}</div></div>"

    # --- NARRATIVE CLUSTERS INDEX ---
    summary_md += "## 📁 Narrative Clusters\n\n"
    summary_html += f"<h2 style='color: #202124; font-size: 20px; border-bottom: 2px solid #1A73E8; padding-bottom: 10px; margin-bottom: 20px; font-weight: bold;'>📁 Narrative Clusters</h2>"
    
    for cluster in cluster_metrics:
        crowd_flag = " ⚠️ **[CROWDED]**" if cluster['is_crowded'] else ""
        summary_md += f"### {cluster['name']}{crowd_flag}\n"
        summary_md += f"**Description:** {cluster['description']} | **Bias:** `{cluster['bias']}`\n"
        summary_md += f"- **Dominance:** {cluster['size']} channels | **Avg Signal:** {cluster['avg_signal']:.1f}/10\n"
        
        summary_html += f"<div style='margin-bottom: 25px; border: 1px solid #eee; border-radius: 8px; overflow: hidden;'>"
        header_bg = "#fde8e8" if cluster['is_crowded'] else "#f8f9fa"
        summary_html += f"<div style='background: {header_bg}; padding: 12px; border-bottom: 1px solid #eee;'>"
        summary_html += f"<div style='font-weight: bold; font-size: 15px; color: #1A73E8;'>{cluster['name']}</div>"
        summary_html += f"<div style='font-size: 13px; color: #3C4043; margin-bottom: 4px;'>{cluster['description']}</div>"
        
        bias_color = {"constructive": "#34A853", "defensive": "#EA4335", "counter-consensus": "#F9AB00", "neutral": "#70757A"}.get(cluster['bias'].lower(), "#70757A")
        summary_html += f"<span style='background: {bias_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; text-transform: uppercase;'>{cluster['bias']}</span>"
        
        if cluster['is_crowded']:
            summary_html += " <span style='background: #EA4335; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 10px;'>CROWDED TRADE</span>"
        summary_html += f"<div style='font-size: 12px; color: #70757A; margin-top: 4px;'>{cluster['size']} channels • Avg Signal: {cluster['avg_signal']:.1f}/10</div></div>"
        
        summary_html += "<table width='100%' style='border-collapse: collapse;'>"
        for v, _ in formatted_briefs:
            if v.get('cluster_id') == cluster['id']:
                b = v.get('brief', {})
                summary_md += f"- **{b.get('channel')}**: [{b.get('episode_title')}]({v.get('url')})\n"
                
                summary_html += "<tr><td style='padding: 10px; border-bottom: 1px solid #f1f3f4;'>"
                summary_html += f"<div style='font-size: 14px;'><b style='color: #202124;'>{b.get('channel')}:</b> <a href='{v.get('url')}' style='color: #1155CC; text-decoration: none;'>{b.get('episode_title')}</a></div>"
                summary_html += f"<div style='font-size: 13px; color: #3C4043; margin-top: 4px;'>{b.get('one_line_summary')}</div>"
                summary_html += "</td></tr>"
        summary_html += "</table></div>"
        
    summary_md += "\n---\n\n"
    summary_html += "<hr style='border: 0; border-top: 1px solid #eee; margin: 40px 0;' />"

    # --- DETAILED BRIEFS SECTION ---
    final_md = summary_md
    final_html = summary_html
    
    for _, (md, html_comp) in formatted_briefs: # Iterate over formatted_briefs to get formatted strings
        final_md += md
        final_html += html_comp
        
    final_html += "</body></html>"
    
    with open(md_filename, "w") as f:
        f.write(final_md)
        
    # Save JSON briefs for the dashboard to enable advanced visualizations
    json_filename = os.path.join("briefs", f"{date_str}.json")
    
    enriched_briefs = []
    for v in processed_queue:
        if v.get('brief'):
            b = v.get('brief')
            b['thumbnail'] = v.get('thumbnail', '')
            b['video_url'] = v.get('url', '')
            b['cluster_id'] = v.get('cluster_id', -1)
            enriched_briefs.append(b)

    final_data = {
        "date": date_str,
        "meta_summary": meta_summary,
        "briefs": enriched_briefs,
        "clusters": cluster_metrics
    }

    with open(json_filename, "w") as f:
        json.dump(final_data, f, indent=2)
    elapsed = time.time() - start_time
    logger.info(f"Summarization complete. {len(processed_queue)} briefs written to {md_filename} (Time: {elapsed:.1f}s)")
    
    send_email = str(os.getenv("SEND_EMAIL", "false")).lower() == "true"
    if send_email:
        send_email_digest(final_html, date_str)
        
    # Clear queue after successful processing
    with open(queue_path, "w") as f:
        json.dump([], f)
    logger.info("Queue cleared.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_summarization()
