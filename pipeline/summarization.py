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
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
GENAI_CLIENT = None
CONFIG_PATH = "config.json"

def load_config():
    """Loads global configuration for thresholds and weights."""
    if not os.path.exists(CONFIG_PATH):
        # Fallback defaults matches config.json
        return {
            "meta": {
                "generation": {"min_videos": 5, "min_clusters": 2, "convergence_threshold": 0.65, "threshold_mode": "fixed", "history_window_days": 30},
                "convergence": {"weights": {"mean_similarity": 0.6, "std_dev_penalty": 0.2, "weighted_dominance": 0.2}, "min_cluster_size": 2}
            },
            "clustering": {"percentile": 85, "crowding_percentile": 75}
        }
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

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
    config = load_config()
    framing_blacklist = config.get("clustering", {}).get("framing_blacklist", [])
    
    template = (
        "You are a Senior Intelligence Analyst. Generate a clinical, signal-dense Intelligence Brief.\n\n"
        "### 🛡️ SOURCE-OF-TRUTH ISOLATION:\n"
        "The ATTACHED AUDIO is your ONLY source. IGNORE channel context, themes, or external knowledge.\n\n"
        "### 🚫 PROHIBITED FRAMING (CRITICAL):\n"
        f"DO NOT use recap phrasing or host framing: {', '.join(framing_blacklist)}.\n"
        "STRICTLY FORBIDDEN: 'This episode discusses', 'The host explains', 'The speaker says'.\n\n"
        "### BRIEFING ARCHITECTURE:\n"
        "- ONE LINE SUMMARY: Distill the DIRECT narrative pressure or signal. No context padding.\n"
        "- CORE CLAIMS: Specific, quantified claims or projections. Extract ONLY differentiated signal.\n"
        "- SIGNAL STRENGTH: Score 1-10 (uniqueness/actionability).\n"
        "- THEMES: lowercase short tags.\n"
        "- POSITIONING IMPLICATION: Strategic consequence. Must be clinical/domain-agnostic.\n"
        "- TIME HORIZON: short | medium | long.\n\n"
        "### CONSTRAINTS:\n"
        "1. TONE: Institutional Strategy Brief. Zero hype.\n"
        "2. INCREMENTAL SIGNAL ONLY: Every point must answer 'What is the specific, new signal here?'\n"
        "3. OUTPUT: Valid JSON matching the schema.\n\n"
        "Title: {title}\nChannel: {channel}\nToday: {today}\n\n"
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
            result = json.loads(response.text)
            # Gemini sometimes wraps the response in an array — unwrap it
            if isinstance(result, list):
                if len(result) > 0:
                    result = result[0]
                else:
                    logger.error(f"Empty JSON array returned for {video['title']}")
                    return None
            return result
        except (json.JSONDecodeError, Exception) as e:
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
        "You are a Lead Intelligence Strategist. Produce a DENSE, QUANTIFIED Regime Detection Report.\n"
        "### STYLE GUIDE:\n"
        "- NO qualitative filler. Clinical, institutional tone.\n"
        "- MAX signal density. MIN words.\n"
        "- USE numerical backing for every assertion."
    )
    
    user_content = (
        "### DATASET DIAGNOSTICS:\n"
        f"Total Assets: {total_assets}\n\n"
        "### NARRATIVE CLUSTERS:\n"
        f"{clusters_input}\n\n"
        "### REQUIRED OUTPUT:\n"
        "1. DOMINANT NARRATIVE: Provide a 1-2 sentence clinical description of the sector regime (highest D_adj and convergence-weighted theme).\n"
        "2. QUANTIFIED DOMINANCE: Identify top cluster by % dominance. State Strength and Coherence.\n"
        "3. NARRATIVE CONVERGENCE: Evaluate degree of agreement across clusters.\n"
        "4. INTRA-CLUSTER FRACTURES: Identify low-coherence clusters or opposing claims.\n"
        "5. DATASET-DERIVED TRIGGERS: 2-3 specific catalysts extracted ONLY from CLAIMS.\n\n"
        "STRICT CONSTRAINTS:\n"
        "- Be 20% shorter than a standard summary.\n"
        "- Tone: Institutional intelligence report."
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

def calculate_convergence_score(clusters_data: List[Dict[str, Any]], total_videos: int, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes Narrative Convergence using stabilized weighted formula:
    convergence = (w1 * μ) - (w2 * σ) + (w3 * D_adj)
    """
    if not clusters_data:
        return {"score": 0.0, "mu": 0.0, "sigma": 0.0, "d_adj": 0.0}

    conv_cfg = config.get("meta", {}).get("convergence", {})
    weights = conv_cfg.get("weights", {})
    min_size = conv_cfg.get("min_cluster_size", 2)

    # Filter clusters by minimum size for convergence stability
    valid_clusters = [c for c in clusters_data if c['size'] >= min_size]
    if len(valid_clusters) < 2:
        return {"score": 0.0, "mu": 0.0, "sigma": 0.0, "d_adj": 0.0}

    # Step 1: Collect Centroids
    centroids = []
    for c in valid_clusters:
        # Re-derive centroid from briefs (assuming they have embeddings or we calculate on the fly)
        # For efficiency, we assume the clustering pass already stored this or we compute it
        cluster_embeddings = [np.array(b['embedding']) for b in c.get('brief_data', []) if 'embedding' in b]
        if cluster_embeddings:
            centroid = np.mean(cluster_embeddings, axis=0)
            # Normalize for cosine similarity (dot product on normalized = cosine)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroids.append(centroid / norm)

    if len(centroids) < 2:
        return {"score": 0.0, "mu": 0.0, "sigma": 0.0, "d_adj": 0.0}

    # Step 2: Centroid Similarity Matrix
    centroids = np.array(centroids)
    sim_matrix = cosine_similarity(centroids)
    
    # Extract Upper Triangle (excluding diagonal)
    mask = np.triu(np.ones(sim_matrix.shape), k=1).astype(bool)
    pairs = sim_matrix[mask]

    # Step 3: Compute Metrics
    mu = float(np.mean(pairs))
    sigma = float(np.std(pairs))
    
    # Dominance Weighting (D_adj = largest_cluster / total_videos * mu)
    largest_cluster = max([c['size'] for c in valid_clusters])
    d_adj = (largest_cluster / total_videos) * mu

    # Step 4: Final Stabilized Formula
    score = (
        weights.get("mean_similarity", 0.6) * mu -
        weights.get("std_dev_penalty", 0.2) * sigma +
        weights.get("weighted_dominance", 0.2) * d_adj
    )

    return {
        "score": max(0.0, min(1.0, score)),
        "mu": mu,
        "sigma": sigma,
        "d_adj": d_adj
    }

def update_convergence_history(category: str, score: float, window_days: int):
    """Logs convergence score to history and maintains rolling window."""
    history_path = "data/convergence_history.json"
    os.makedirs("data", exist_ok=True)
    
    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except:
            history = {}

    if category not in history:
        history[category] = []
    
    history[category].append({
        "timestamp": datetime.now().isoformat(),
        "score": score
    })

    # Simple cleanup (keeping for N entries as proxy for days since runs are periodic)
    # Ideally we'd filter by timestamp, but for simplicity we keep last 50 runs per category
    history[category] = history[category][-50:]

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

def calculate_percentile(category: str, current_score: float) -> float:
    """Calculates percentile of current score relative to category history."""
    history_path = "data/convergence_history.json"
    if not os.path.exists(history_path):
        return 0.0
        
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
        scores = [h['score'] for h in history.get(category, [])]
        if not scores:
            return 0.0
            
        return float(np.mean(np.array(scores) <= current_score) * 100)
    except:
        return 0.0
def generate_cluster_label(cluster_briefs, llm):
    """
    Generates a dynamic narrative pressure name, description, and positioning bias.
    """
    config = load_config()
    blacklist = config.get("clustering", {}).get("generic_label_blacklist", [])
    pressure_verbs = config.get("clustering", {}).get("narrative_pressure_verbs", [])
    
    # Access brief data correctly
    context = "\n".join([f"- {b.get('brief', {}).get('one_line_summary', 'N/A')} (Themes: {', '.join(b.get('brief', {}).get('themes', []))})" for b in cluster_briefs[:3]])
    
    prompt = (
        "You are a Senior Macro Intelligence Strategist. Generate a human-readable NARRATIVE PRESSURE label for a cluster.\n\n"
        f"### CLUSTER CONTENT:\n{context}\n\n"
        "### LABEL LOGIC PRIORITY:\n"
        "1. Detect repeated directional pressure (movement, tension, or conflict).\n"
        f"2. Use convergence verbs for intensity: {', '.join(pressure_verbs)}.\n"
        "3. Combine the highest-weighted Entity with the Pressure Mechanism.\n\n"
        "### CONSTRAINTS:\n"
        "- MAX 8 WORDS.\n"
        "- NO qualitative filler or generic nouns (e.g., 'analysis', 'overview', 'discussion').\n"
        f"- AVOID generic suffixes: {', '.join(blacklist)}.\n"
        "- STYLE: Institutional, clinical, directional.\n"
        "- NO reference to specific channels or hosts.\n"
        "- FORMAT: JSON only.\n\n"
        "### OUTPUT FORMAT:\n"
        "{{\n"
        "  \"cluster_name\": \"...\",\n"
        "  \"description\": \"One sentence answering: What incremental signal does this narrative add?\",\n"
        "  \"positioning_bias\": \"constructive|defensive|neutral|counter-consensus\"\n"
        "}}\n"
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        import json
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
    date_str = datetime.now().strftime("%Y-%m-%d")
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

    # --- PHASE 3: CATEGORY-LEVEL META SYNTHESIS ---
    logger.info(f"--- Phase 3: Category Synthesis ({len(processed_queue)} assets across categories) ---")
    config = load_config()
    summary_md = f"# TheBrief Daily Dispatch - {date_str}\n\n"
    summary_html = f"<html><body style='font-family: Arial, Helvetica, sans-serif; color: #202124; background-color: #ffffff; padding: 20px;'>"
    summary_html += f"<h1 style='color: #202124; font-size: 24px; margin-bottom: 5px;'>TheBrief Daily Dispatch</h1>"
    summary_html += f"<div style='color: #70757A; margin-bottom: 25px; font-size: 14px;'>Intelligence Report: {date_str}</div>"

    try:
        # A. Group Clusters by Category
        from collections import defaultdict
        category_map = defaultdict(list)
        for v in processed_queue:
            cat = v.get("category", "Other")
            category_map[cat].append(v)

        total_assets = len(processed_queue)
        meta_cfg = config.get("meta", {}).get("generation", {})
        category_intelligence = {}

        logger.info(f"Analyzing {len(category_map)} categories: {', '.join(category_map.keys())}")
        for category, cat_videos in category_map.items():
            logger.info(f"Processing Synthesis for sector: {category} ({len(cat_videos)} assets)...")
            
            # B. Perform semantic clustering WITHIN the category (User constraint: NO other grouping)
            current_percentile = config.get("clustering", {}).get("percentile", 85)
            linkage_mode = os.getenv("CLUSTERING_LINKAGE", "complete")
            cat_videos = perform_semantic_clustering(
                cat_videos, 
                percentile=current_percentile,
                linkage=linkage_mode
            )

            # C. Identify clusters present in this category
            cat_clusters = defaultdict(list)
            for v in cat_videos:
                if v.get('cluster_id') != -1:
                    cat_clusters[v['cluster_id']].append(v)
            
            # --- CLUSTER DEDUPLICATION ---
            dedup_threshold = config.get("clustering", {}).get("deduplication_threshold", 0.88)
            cluster_centroids = {}
            for c_id, briefs in cat_clusters.items():
                embeddings = [b.get('embedding') for b in briefs if b.get('embedding')]
                if embeddings:
                    cluster_centroids[c_id] = np.mean(embeddings, axis=0)
            
            merged_ids = {}
            active_ids = list(cluster_centroids.keys())
            for i in range(len(active_ids)):
                id_a = active_ids[i]
                if id_a in merged_ids: continue
                for j in range(i + 1, len(active_ids)):
                    id_b = active_ids[j]
                    if id_b in merged_ids: continue
                    
                    sim = cosine_similarity([cluster_centroids[id_a]], [cluster_centroids[id_b]])[0][0]
                    if sim > dedup_threshold:
                        logger.info(f"🔄 Merging redundant clusters {id_a} and {id_b} (Sim: {sim:.3f})")
                        merged_ids[id_b] = id_a
            
            # Re-apply merges to cat_clusters
            final_clusters = defaultdict(list)
            for c_id, briefs in cat_clusters.items():
                target_id = merged_ids.get(c_id, c_id)
                final_clusters[target_id].extend(briefs)
            
            cat_cluster_metrics = []
            for c_id, briefs in final_clusters.items():
                avg_signal = np.mean([b.get('brief', {}).get('signal_strength', 5) for b in briefs])
                avg_coherence = np.mean([b.get('cluster_coherence', 1.0) for b in briefs])
                cluster_strength = np.mean([b.get('cluster_strength', 0) for b in briefs])
                
                label_data = generate_cluster_label(briefs, llm)
                
                # Regime State logic per cluster (Emerging -> Dominant -> Decaying)
                regime_labels = meta_cfg.get("narrative_regime_labels", ["Emerging", "Dominant", "Decaying"])
                if len(briefs) >= 4:
                    c_regime = regime_labels[2] # Dominant
                elif len(briefs) >= 2:
                    c_regime = regime_labels[1] # Accelerating/Dominant
                else:
                    c_regime = regime_labels[0] # Emerging
                
                # Fetch centroid for cross-cluster correlation
                centroid = cluster_centroids.get(c_id)
                
                cat_cluster_metrics.append({
                    "id": c_id,
                    "name": label_data.get("cluster_name"),
                    "description": label_data.get("description"),
                    "bias": label_data.get("positioning_bias"),
                    "regime": c_regime,
                    "size": len(briefs),
                    "strength": float(cluster_strength),
                    "coherence": float(avg_coherence),
                    "channels": list(set([b.get('brief', {}).get('channel', b.get('channel', 'Unknown')) for b in briefs])),
                    "avg_signal": float(avg_signal),
                    "briefs": [b['brief'] for b in briefs],
                    "centroid": centroid.tolist() if centroid is not None else None
                })
            
            # Sort by size for hierarchy
            cat_cluster_metrics = sorted(cat_cluster_metrics, key=lambda x: x['size'], reverse=True)
            
            # D. Separate Headlines from Peripheral Signals with folding logic
            min_headline_size = meta_cfg.get("min_cluster_size_for_headline", 2)
            p_threshold = meta_cfg.get("peripheral_strength_threshold", 0.35)
            
            # Calculate Convergence early for folding logic
            conv_metrics = calculate_convergence_score(cat_cluster_metrics, len(cat_videos), config)
            convergence_score = conv_metrics['score']
            
            headline_clusters = []
            peripheral_clusters = []
            
            for c in cat_cluster_metrics:
                is_weak = c['size'] < min_headline_size
                # If sector is highly aligned, we fold small clusters into primary view
                if is_weak and convergence_score < 0.4: # Low alignment -> hide weak signals
                    peripheral_clusters.append(c)
                else:
                    headline_clusters.append(c)

            # Log to history
            update_convergence_history(category, convergence_score, meta_cfg.get("history_window_days", 30))
            
            # Regime State Logic
            regime_thresholds = meta_cfg.get("regime_thresholds", {})
            regime_state = "Neutral"
            for state, t_val in sorted(regime_thresholds.items(), key=lambda x: x[1], reverse=True):
                if convergence_score >= t_val:
                    regime_state = state
                    break

            # Check Thresholds for Summary Generation
            threshold = meta_cfg.get("convergence_threshold", 0.65)
            if meta_cfg.get("threshold_mode") == "percentile":
                p = calculate_percentile(category, convergence_score)
                should_generate = p >= threshold 
            else:
                should_generate = convergence_score >= threshold
            
            # Additional guards
            if len(cat_videos) < meta_cfg.get("min_videos", 5): should_generate = False
            if not final_clusters: should_generate = False

            cat_meta_summary = ""
            if should_generate:
                logger.info(f"✨ Generating Category Meta Summary for {category} (Score: {convergence_score:.2f})")
                cat_meta_summary = generate_meta_summary(cat_cluster_metrics, len(cat_videos), llm)

            # Store for dashboard (CONTRACT-FIRST SCHEMA)
            category_intelligence[category] = {
                "name": category,
                "meta_summary": cat_meta_summary,
                "convergence_score": float(convergence_score),
                "percentile": float(calculate_percentile(category, convergence_score)) if meta_cfg.get("threshold_mode") == "percentile" else 0.0,
                "regime_state": regime_state,
                "clusters": cat_cluster_metrics,
                "should_generate": should_generate
            }

            # Build Output for this category
            summary_md += f"## Sector: {category}\n"
            summary_md += f"**Convergence:** `{convergence_score:.2f}` | **Regime:** `{regime_state}`\n\n"
            
            if cat_meta_summary:
                summary_md += f"> ### 🧠 Dominant Narrative\n> {cat_meta_summary}\n\n"
                summary_html += f"<div style='background: #1A73E8; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>"
                summary_html += f"<h3 style='margin-top: 0; font-size: 16px;'>Sector Intelligence: {category} ({regime_state})</h3>"
                summary_html += f"<div style='font-size: 14px;'>{cat_meta_summary.replace(chr(10), '<br/>')}</div></div>"
            
            # Render Headlines
            for cluster in headline_clusters:
                summary_md += f"### {cluster['name']}\n"
                summary_md += f"**Description:** {cluster['description']} | **Bias:** `{cluster['bias']}`\n"
                summary_md += f"- **Dominance:** {cluster['size']} channels | **Avg Signal:** {cluster['avg_signal']:.1f}/10\n"
                
                summary_html += f"<div style='margin-bottom: 15px; border-left: 4px solid #1A73E8; padding-left: 10px;'>"
                summary_html += f"<div style='font-weight: bold; color: #1A73E8;'>{cluster['name']}</div>"
                summary_html += f"<div style='font-size: 13px;'>{cluster['description']}</div>"
                
                for b in cluster['briefs']:
                    summary_md += f"- **{b.get('channel')}**: {b.get('one_line_summary')}\n"
                    summary_html += f"<div style='font-size: 12px; color: #70757A;'>• <b>{b.get('channel')}:</b> {b.get('one_line_summary')}</div>"
                summary_html += "</div>"

            # Render Peripheral Signals
            if peripheral_clusters:
                p_limit = meta_cfg.get("peripheral_render_limit", 5)
                summary_md += "\n#### 📡 Peripheral Signals\n"
                summary_html += "<div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
                summary_html += "<div style='font-weight: bold; font-size: 13px; color: #70757A;'>📡 Peripheral Signals</div>"
                
                for i, c in enumerate(peripheral_clusters[:p_limit]):
                    for b in c['briefs']:
                        summary_md += f"- {b.get('one_line_summary')} ({b.get('channel')})\n"
                        summary_html += f"<div style='font-size: 11px; color: #70757A;'>• {b.get('one_line_summary')} ({b.get('channel')})</div>"
                summary_html += "</div>"
            
            logger.info(f"✅ Synthesis complete for {category}")

        # --- CROSS-CLUSTER CORRELATION PASS ---
        from metrics.cross_cluster_correlation import CrossClusterCorrelation
        correlator = CrossClusterCorrelation(config)
        correlations = correlator.detect(category_intelligence)
        
        for corr in correlations:
            src = corr["source"]
            tgt = corr["target"]
            alert_text = f"\n⚡ Convergence Alert: Overlap detected with [{tgt['name']}] in {tgt['category']} ({corr['similarity']:.2f} sim)"
            
            # Find the cluster in category_intelligence and append alert
            cat_intel = category_intelligence.get(src["category"], {})
            for cluster in cat_intel.get("clusters", []):
                if cluster["id"] == src["id"]:
                    cluster["description"] += alert_text
                    break

        # Cleanup: Remove centroids from final JSON data contract (internal use only)
        for cat_data in category_intelligence.values():
            for cluster in cat_data.get("clusters", []):
                if "centroid" in cluster:
                    del cluster["centroid"]

    except Exception as e:
        logger.error(f"CRITICAL ERROR in Synthesis Phase: {e}", exc_info=True)
        summary_md += "\n\n> [!ERROR]\n> Category Synthesis failed due to a system error. Detailed briefs are available below.\n\n"

    # --- FINAL OUTPUT COMPILATION ---
    md_filename = os.path.join("briefs", f"{date_str}.md")
    os.makedirs("briefs", exist_ok=True)

    # Detailed Briefs Section
    summary_md += "\n---\n## Detailed Intelligence Briefs\n\n"
    summary_html += "<hr style='border: 0; border-top: 1px solid #eee; margin: 40px 0;' /><h2 style='color: #202124;'>Detailed Intelligence Briefs</h2>"
    
    for v_comp, (md_comp, html_comp) in formatted_briefs:
        summary_md += md_comp
        summary_html += html_comp

    summary_html += "</body></html>"

    with open(md_filename, "w") as f:
        f.write(summary_md)
        
    # Save JSON briefs for the dashboard
    json_filename = os.path.join("briefs", f"{date_str}.json")
    
    enriched_briefs = []
    for v in processed_queue:
        if v.get('brief'):
            b = v.get('brief')
            b['thumbnail'] = v.get('thumbnail', '')
            b['video_url'] = v.get('url', '')
            b['cluster_id'] = v.get('cluster_id', -1)
            b['category'] = v.get('category', 'Other')
            enriched_briefs.append(b)

    final_data = {
        "date": date_str,
        "briefs": enriched_briefs,
        "category_intelligence": category_intelligence,
        "clusters": [] 
    }

    with open(json_filename, "w") as f:
        json.dump(final_data, f, indent=2)

    # Update processed_videos DB
    db_path = os.path.join("data", "processed_videos.json")
    try:
        from tinydb import TinyDB
        db = TinyDB(db_path)
        for v in processed_queue:
            db.insert({"id": v["id"], "title": v["title"], "processed_at": datetime.now().isoformat()})
    except Exception as e:
        logger.warning(f"Note: Error updating processed_videos db: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Summarization complete. {len(processed_queue)} briefs written to {md_filename} (Time: {elapsed:.1f}s)")
    
    send_email = str(os.getenv("SEND_EMAIL", "false")).lower() == "true"
    if send_email:
        send_email_digest(summary_html, date_str)
        
    # Clear queue after successful processing
    with open(queue_path, "w") as f:
        json.dump([], f)
    logger.info("Queue cleared.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_summarization()
