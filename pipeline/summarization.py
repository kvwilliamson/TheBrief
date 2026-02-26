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
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from google import genai
import time
from pipeline.profiles import get_profile_for_category

logger = logging.getLogger(__name__)

# --- High-Utility Decision Format (vNext) Schemas ---

class ExecutiveUseCase(BaseModel):
    signal_type: str = Field(description="Strictly one of: Sentiment, Thesis, Data-driven, Speculative, Promotional, Mixed")
    positioning_impact: str = Field(description="Strictly one of: No Action, Monitor, Minor Bias, High Conviction Shift")
    time_horizon: str = Field(description="Strictly one of: Short-term, Cyclical, Structural")
    confidence_level: str = Field(description="Strictly one of: Low, Moderate, High")
    incentive_bias: str = Field(description="Strictly: Yes / No / Mild, with a one sentence max explanation")
    consensus_context: str = Field(description="Is this mainstream or fringe? Is it widely discussed?")

class Claim(BaseModel):
    claim: str = Field(description="The core claim or forward projection")
    evidence_cited: str = Field(description="Brief description of the evidence cited")
    evidence_type: str = Field(description="Strictly one of: Anecdotal, Data-backed, Assumed, Historical reference")
    empirical_strength: str = Field(description="Strictly one of: Low, Moderate, High. Measures hard data/cited evidence.")
    speaker_conviction: str = Field(description="Strictly one of: Low, Moderate, High. Measures rhetorical volume/certainty.")

class Mechanism(BaseModel):
    trigger: str = Field(description="The trigger event")
    transmission_path: str = Field(description="How the trigger propagates")
    market_impact: str = Field(description="The ultimate impact")
    secondary_effects: str = Field(description="Any secondary effects")

class BriefSchema(BaseModel):
    episode_title: str = Field(description="Title of the episode")
    channel: str = Field(description="Name of the channel")
    duration_minutes: int = Field(description="Duration in minutes")
    topic_domain: str = Field(description="Broad topic domain")
    podcast_date: str = Field(description="Date episode was published (from context)")
    processing_date: str = Field(description="Date TheBrief processed it (today's date)")
    shelf_life: str = Field(description="Strictly: Short (days-weeks), Medium (weeks-months), Long (structural/multi-year)")
    current_market_context: str = Field(description="Brief snapshot of market at processing date based on current date knowledge")
    executive_use_case: ExecutiveUseCase
    core_claims: List[Claim] = Field(description="Combined core claims and forward projections (Max 6)")
    specifics_extracted: Optional[str] = Field(description="Verbatim numbers/targets. Use 'No explicit targets mentioned' if none or if qualitative topic.")
    weak_links: str = Field(description="Identified failure points in the thesis causal chain")
    counter_consensus: str = Field(description="2-3 bullets on mainstream/institutional alternative view")
    meta_assessment: Optional[str] = Field(description="Framing pattern, conviction level. Max 3 bullets.")
    mechanism: Mechanism = Field(description="Stress-test the logic or describe the framework/pathway.")
    disconfirming_signals: List[str] = Field(description="Max 3 observable disconfirming signals.")
    historical_parallel: Optional[str] = Field(description="Optional comparison to a historical parallel.")
    one_line_summary: str = Field(description="A single sentence summarizes the core thesis.")
    emotional_conviction: str = Field(description="Summary of the speaker's tone and underlying conviction.")
    signal_strength: Optional[int] = Field(default=None, ge=1, le=10, description="Score 1-10 per profile rubric.")
    signal_strength_justification: Optional[str] = Field(description="One sentence referencing rubric and why.")
    novelty: Optional[int] = Field(default=None, ge=1, le=10, description="Score 1-10 per profile rubric.")
    novelty_justification: Optional[str] = Field(description="One sentence referencing rubric and why.")
    tradeability: Optional[int] = Field(default=None, ge=1, le=10, description="Score 1-10 per profile rubric.")
    tradeability_justification: Optional[str] = Field(description="One sentence referencing rubric and why.")
    time_sensitivity: Optional[str] = Field(description="Strictly: Immediate, Monitor, Long-term")
    speaker_context: Optional[str] = Field(description="Known background or financial interest. Max 2 sentences.")
    claim_plausibility: List[str] = Field(description="Per-claim plausibility classification.")
    positioning_risk: Optional[str] = Field(description="Only if financial topic.")
def get_llm():
    model_choice = os.getenv("SUMMARY_MODEL", "gemini").lower()
    
    if model_choice == "gemini":
        try:
            # We must use gemini-2.5-flash for native audio understanding
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
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

def summarize_transcript(video, llm):
    # Start briefing
    
    parser = JsonOutputParser(pydantic_object=BriefSchema)
    
    # 1. Upload Audio to Gemini
    audio_path = video.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Missing audio path for {video['title']}")
        return None
        
    client = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))
    # logger.info(f"Uploading {audio_path}...")
    file_handle = client.files.upload(file=audio_path)
    
    # Wait for processing
    while file_handle.state == "PROCESSING":
        time.sleep(2)
        file_handle = client.files.get(name=file_handle.name)
        
    if file_handle.state == "FAILED":
        logger.error(f"Gemini processing failed for {video['title']}")
        return None
        
    logger.info(f"Generating Brief for {video['title']} ({video['id']})...")
    
    # --- Profile-Driven Customization ---
    category = video.get("category", "Other")
    profile = get_profile_for_category(category)
    features = profile.get("features", {})
    rubric_config = profile.get("rubric", {})
    
    # Construct Calculations Block
    calcs_block = "QUANTITATIVE SCORING — PROFESSIONAL RUBRIC:\nRate items (1-10) using these DERIVED formulas:\n"
    if rubric_config.get("signal"):
        calcs_block += f"1. {rubric_config['signal']}\n"
    if rubric_config.get("tradeability"):
        calcs_block += f"2. {rubric_config['tradeability']}\n"
    calcs_block += "JUSTIFICATION: Provide one sentence per score referencing components of the formula above.\n\n"
    
    # 2. Build the LLM Chain natively passing the file_handle
    today_str = datetime.now().strftime("%Y-%m-%d")
    template = (
        "You are an expert intelligence analyst.\n"
        "Generate a high-utility, educational, and decision-oriented Intelligence Brief directly from the attached audio.\n\n"
        "### BKM CORE INSTRUCTIONS:\n\n"
        "TIMESTAMP & SHELF LIFE:\n"
        "- Record the episode publication date and today's processing date: {today}.\n"
        "- Assign a Shelf Life: Short (days-weeks), Medium (weeks-months), Long (structural).\n"
        "- Provide a brief 'current_market_context' snapshot relative to {today} (assets/trends).\n\n"
    )
    
    if features.get("specifics"):
        template += (
            f"SPECIFICS EXTRACTION ({features['specifics']}):\n"
            "- Extract ALL relevant technical/financial targets or benchmarks verbatim.\n"
            "- CRITICAL: If no specifics were mentioned, state: 'No explicit specifics mentioned.'\n\n"
        )
    
    template += (
        "CLAIM PLAUSIBILITY CHECK — PER CLAIM:\n"
        "For each core claim, provide a plausibility classification.\n"
        "Classifications: VERIFIED, ASSERTED, INFERRED, HEADLINE RISK.\n"
        "- ⚠️ ANTI-HOAXING RULE: Punish precision. Specific quantitative claims (e.g., \"20% target\") MUST remain ASSERTED unless confirmed by public record.\n\n"
        f"{calcs_block}"
        "EVIDENCE DISAMBIGUATION — CRITICAL:\n"
        "- Distinguish between Evidence (empirical) and Conviction (rhetorical).\n"
        "- 'Assumed' or 'Anecdotal' evidence MUST result in 'Low' or 'Moderate' Empirical Strength, even if Speaker Conviction is 'High'.\n\n"
        "INTELLIGENCE PROFILE — CONCISION RULES:\n"
        "- Speaker Context: Maximum 2 sentences. Verifiable facts only. If financial interest exists, state it FIRST.\n"
        "- Meta Assessment: Max 3 bullets on framing, conviction, and incentive bias.\n\n"
        f"MATERIALITY ANCHORING ({features.get('mechanism', 'Mechanism')}):\n"
        "- This section MUST open with a quantified statement or first-principles logic anchor.\n"
        "- Stress-test the underlying logic/architecture; do not just restate the story.\n\n"
        "STRICT CONSTRAINTS:\n"
        "1. Tone: Professional intelligence memo. No hype, no emojis.\n"
        "2. Max word count: 400-600 words.\n"
        "3. Output MUST conform exactly to the JSON schema.\n\n"
        "Video details:\nTitle: {title}\nChannel: {channel}\nDuration: {duration_minutes} minutes\nToday: {today}\n\n"
        "Format instructions:\n{format_instructions}\n\n"
    )
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["title", "channel", "duration_minutes", "today"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    try:
        # Langchain Google GenAI accepts a list of [prompt_text, gemini_file_uri]
        formatted_prompt = prompt.format(
            title=video.get("title", ""),
            channel=video.get("channel", ""),
            duration_minutes=video.get("duration_minutes", 0),
            tags=", ".join(video.get("tags", [])),
            today=today_str
        )
        
        # Invoke native multi-modal model
        response = llm.invoke([formatted_prompt, file_handle.uri])
        
        # Parse the JSON response manually since we bypassed the standard chain
        try:
            result = parser.parse(response.content)
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

def format_markdown(brief: dict, video_url: str = "", thumbnail_url: str = "") -> str:
    title = brief.get('episode_title', 'Unknown Title')
    category = brief.get('topic_domain', 'Other')
    profile = get_profile_for_category(category)
    features = profile.get("features", {})
    
    md = ""
    if video_url:
        md += f"## [{title}]({video_url})\n"
    else:
        md += f"## {title}\n"
        
    if thumbnail_url:
        md += f"![{title}]({thumbnail_url})\n\n"

    # Incentive Bias Banner
    if brief.get('executive_use_case', {}).get('incentive_bias', 'No') != 'No':
        md += f"> ⚠️ **INCENTIVE BIAS FLAGGED:** {brief['executive_use_case']['incentive_bias']}\n\n"

    md += f"**{brief.get('channel', 'Unknown')}** | **Length:** {brief.get('duration_minutes', 0)} min | **Market Context:** {brief.get('current_market_context', 'N/A')}\n"
    md += f"**Published:** {brief.get('podcast_date', 'N/A')} | **Processed:** {brief.get('processing_date', 'N/A')} | **Shelf Life:** {brief.get('shelf_life', 'N/A')}\n\n"
    
    md += "### Quick-Scan Metrics\n"
    md += f"- **Signal Strength:** {brief.get('signal_strength', 0)}/10 — {brief.get('signal_strength_justification', 'N/A')}\n"
    if features.get("novelty"):
        md += f"- **Novelty:** {brief.get('novelty', 0)}/10 — {brief.get('novelty_justification', 'N/A')}\n"
    if features.get("tradeability"):
        md += f"- **Tradeability:** {brief.get('tradeability', 0)}/10 — {brief.get('tradeability_justification', 'N/A')}\n"
    md += f"- **Time Sensitivity:** {brief.get('time_sensitivity', 'N/A')}\n\n"

    md += f"**Thesis:** {brief.get('one_line_summary')}\n\n"

    md += "### Intelligence Profile\n"
    md += f"- **Speaker:** {brief.get('speaker_context')}\n"
    md += f"- **Meta Assessment:** {brief.get('meta_assessment')}\n"
    md += f"- **Emotional Tone:** {brief.get('emotional_conviction')}\n\n"

    claims = brief.get('core_claims', [])
    if claims:
        md += "### Core Claims\n"
        for claim in claims[:6]:
            md += f"- **{claim.get('claim')}**\n"
            md += f"  *Evidence:* {claim.get('evidence_cited')} ({claim.get('evidence_type')} | Empirical: {claim.get('empirical_strength')} | Conviction: {claim.get('speaker_conviction')})\n\n"

    md += "### Weak Links & Failures\n"
    md += f"{brief.get('weak_links')}\n\n"

    md += "### Claim Plausibility Check\n"
    plausibility = brief.get('claim_plausibility', [])
    if isinstance(plausibility, list):
        for p in plausibility:
            md += f"- {p}\n"
    else:
        md += f"{plausibility}\n"
    md += "\n"

    if features.get("specifics"):
        md += f"### {features['specifics']}\n"
        md += f"{brief.get('specifics_extracted')}\n\n"

    mech = brief.get('mechanism', {})
    if mech:
        md += f"### {features.get('mechanism', 'Mechanism')} (Logic Stress-Test)\n"
        md += f"- **Trigger:** {mech.get('trigger')}\n"
        md += f"- **Transmission:** {mech.get('transmission_path')}\n"
        md += f"- **Impact Target:** {mech.get('market_impact')}\n"
        md += f"- **Secondary Effects:** {mech.get('secondary_effects')}\n\n"

    md += "### Counter-Consensus View\n"
    cc = brief.get('counter_consensus', '')
    if isinstance(cc, list): cc = "\n".join([f"- {i}" for i in cc])
    md += f"{cc}\n\n"

    if features.get("signals"):
        signals = brief.get('disconfirming_signals', [])
        if signals:
            md += "### Disconfirming Signals to Watch\n"
            for sig in signals[:3]:
                md += f"- {sig}\n"
            md += "\n"

    hist = brief.get('historical_parallel')
    if hist:
        md += "### Historical Parallel\n"
        md += f"{hist}\n\n"

    md += "---\n\n"
    return md

def format_html(brief: dict, video_url: str = "", thumbnail_url: str = "") -> str:
    title = brief.get('episode_title', 'Unknown Title')
    category = brief.get('topic_domain', 'Other')
    profile = get_profile_for_category(category)
    features = profile.get("features", {})
    
    html = f"<div style='border: 1px solid #333; padding: 20px; border-radius: 8px; background-color: #1a1a1a; color: #eee; margin-bottom: 30px;'>"
    
    if video_url:
        html += f"<h2><a href='{video_url}' style='text-decoration: none; color: #00d1b2;'>{title}</a></h2>"
    else:
        html += f"<h2>{title}</h2>"
    
    if thumbnail_url:
        html += f"<div style='margin-bottom: 15px;'><img src='{thumbnail_url}' width='320' style='border-radius: 8px;' alt='{title}'/></div>"

    # Incentive Bias Banner
    incentive = brief.get('executive_use_case', {}).get('incentive_bias', 'No')
    if incentive != 'No':
        html += f"<div style='background:#ff3860; color:white; padding:15px; border-radius:5px; margin:15px 0;'>⚠️ <b>INCENTIVE BIAS FLAGGED:</b> {incentive}</div>"

    html += f"<p style='margin-bottom: 5px;'><strong>{brief.get('channel', 'Unknown')}</strong> | <strong>Length:</strong> {brief.get('duration_minutes', 0)} min | <strong>Market Context:</strong> {brief.get('current_market_context', 'N/A')}</p>"
    html += f"<p style='color: #888; font-size: 0.9em;'><strong>Published:</strong> {brief.get('podcast_date')} | <strong>Processed:</strong> {brief.get('processing_date')} | <strong>Shelf Life:</strong> {brief.get('shelf_life')}</p>"
    
    # Intelligence Metrics Table (Robust for Email Clients)
    # Determine columns
    cols = []
    if features.get("signal_strength", True): cols.append(("Signal Force", brief.get("signal_strength"), "#00d1b2"))
    if features.get("novelty"): cols.append(("Novelty", brief.get("novelty"), "#ffdd57"))
    if features.get("tradeability"): cols.append(("Tradeability", brief.get("tradeability"), "#48c774"))
    cols.append(("Horizon", brief.get("time_sensitivity"), "#eee"))
    
    width_pct = 100 // len(cols) if cols else 100
    
    html += "<table width='100%' style='border-top: 1px solid #333; border-bottom: 1px solid #333; margin: 15px 0; border-collapse: collapse;'>"
    html += "<tr>"
    for label, _, _ in cols:
        html += f"<th width='{width_pct}%' style='padding: 10px 5px 0; text-align: center; color: #888; font-size: 10px; text-transform: uppercase;'>{label}</th>"
    html += "</tr>"
    html += "<tr>"
    for label, val, color in cols:
        val_str = f"{val}/10" if label != "Horizon" else str(val)
        font_size = "22px" if label != "Horizon" else "16px"
        html += f"<td style='padding: 5px 10px 15px; text-align: center; font-size: {font_size}; font-weight: bold; color: {color};'>{val_str}</td>"
    html += "</tr>"
    html += "</table>"

    html += f"<p style='background: #252525; padding: 10px; border-left: 4px solid #00d1b2;'><b>Thesis:</b> {brief.get('one_line_summary')}</p>"

    html += f"<h3>Intelligence Profile</h3>"
    html += f"<p><b>Speaker:</b> {brief.get('speaker_context')}<br/>"
    html += f"<b>Meta:</b> {brief.get('meta_assessment')}<br/>"
    html += f"<b>Tone:</b> {brief.get('emotional_conviction')}</p>"

    claims = brief.get('core_claims', [])
    if claims:
        html += "<h3>Core Claims</h3>"
        for claim in claims[:6]:
            html += f"<div style='margin-bottom:10px;'><strong>{claim.get('claim')}</strong><br/>"
            html += f"<span style='color: #aaa; font-size: 0.85em;'>Evidence: {claim.get('evidence_cited')} ({claim.get('evidence_type')} | Empirical: {claim.get('empirical_strength')} | Conviction: {claim.get('speaker_conviction')})</span></div>"

    html += f"<h3>Weak Links & Failures</h3><p style='color: #ff3860;'>{brief.get('weak_links')}</p>"
    html += f"<h3>Claim Plausibility</h3>"
    plausibility = brief.get('claim_plausibility', [])
    if isinstance(plausibility, list):
        html += "<ul>"
        for p in plausibility:
            html += f"<li style='margin-bottom: 5px;'>{p}</li>"
        html += "</ul>"
    else:
        html += f"<p style='background: #333; padding: 10px; border-left: 4px solid #ffdd57;'>{plausibility}</p>"
    
    hist = brief.get('historical_parallel')
    if hist:
        html += f"<h3>Historical Parallel</h3><p>{hist}</p>"

    if features.get("specifics"):
        html += f"<h3>{features['specifics']}</h3><pre style='background: #000; padding: 10px; color: #00ff00; font-family: monospace;'>{brief.get('specifics_extracted')}</pre>"

    mech = brief.get('mechanism', {})
    if mech:
        html += f"<h3>{features.get('mechanism', 'Mechanism')} (Logic Stress-Test)</h3><ul>"
        html += f"<li><strong>Trigger:</strong> {mech.get('trigger')}</li>"
        html += f"<li><strong>Transmission:</strong> {mech.get('transmission_path')}</li>"
        html += f"<li><strong>Impact:</strong> {mech.get('market_impact')}</li>"
        html += f"<li><strong>Secondary:</strong> {mech.get('secondary_effects')}</li></ul>"

    cc = brief.get('counter_consensus', 'N/A')
    if isinstance(cc, list): cc = "<br/>".join([f"• {i}" for i in cc])
    html += f"<h3>Counter-Consensus</h3><p>{cc}</p>"

    if features.get("signals"):
        signals = brief.get('disconfirming_signals', [])
        if signals:
            html += "<h3>Disconfirming Signals</h3><ul>"
            for sig in signals[:3]:
                html += f"<li>{sig}</li>"
            html += "</ul>"

    html += "</div>"
    return html

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
            
        # Connect to server
        server = smtplib.SMTP(host, port)
        server.starttls()
        smtp_user = os.getenv("SMTP_USER", email_from)
        server.login(smtp_user, smtp_password)
        
        # Send separate emails to each recipient for maximum reliability
        recipients = [email.strip() for email in email_to.split(",")]
        for recipient in recipients:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"TheBrief Daily Digest - {date_str}"
            msg["From"] = email_from
            msg["To"] = recipient
            
            part = MIMEText(html_content, "html")
            msg.attach(part)
            
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
    briefs_content = []
    start_time = time.time()

    def process_video_summary(video):
        # Transcripts no longer required; we pass audio directly
        brief = summarize_transcript(video, llm)
        if brief:
            video["brief"] = brief
            # Generate markdown and HTML components
            # Generate markdown and HTML components with URL
            # Generate markdown and HTML components with URL and Thumbnail
            md = format_markdown(brief, video.get('url', ''), video.get('thumbnail', ''))
            html = format_html(brief, video.get('url', ''), video.get('thumbnail', ''))
            
            logger.info(f"✅ Brief successfully generated for {video['title']}")
            return video, (md, html)
        return None, None

    # Run summarization concurrently
    from collections import defaultdict
    grouped_briefs = defaultdict(list)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_video = {executor.submit(process_video_summary, video): video for video in queue}
        for future in concurrent.futures.as_completed(future_to_video):
            v_res, format_res = future.result()
            if v_res and format_res:
                processed_queue.append(v_res)
                cat = v_res.get('category', 'Other')
                grouped_briefs[cat].append((v_res, format_res))
                
    if not processed_queue:
        logger.warning("No briefs were successfully generated.")
        return []
        
    # Update processed_videos DB in the main thread to avoid concurrency issues
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
    
    # Calculate Aggregate Stats
    total_videos = len(processed_queue)
    total_time = sum(v.get('duration_minutes', 0) for v in processed_queue)
    
    # Build the "At a Glance" Summary Section
    summary_md = f"# TheBrief Daily Dispatch - {date_str}\n\n"
    summary_md += f"### 📊 At a Glance\n"
    summary_md += f"- **Total Intelligence Assets:** {total_videos} videos\n"
    summary_md += f"- **Total Subject Time:** {total_time:.1f} minutes\n\n"
    
    summary_html = f"<html><body style='font-family: sans-serif; color: #333;'>"
    summary_html += f"<h1>TheBrief Daily Dispatch - {date_str}</h1>"
    summary_html += f"<h3>📊 At a Glance</h3>"
    summary_html += f"<ul><li><strong>Total Intelligence Assets:</strong> {total_videos} videos</li>"
    summary_html += f"<li><strong>Total Subject Time:</strong> {total_time:.1f} minutes</li></ul>"

    # Define Category Order (Matching app.py for consistency)
    cat_order = [
        "General Financial Investing and Speculation",
        "Precious Metals",
        "Artificial Intelligence",
        "Health and Nutrition",
        "Philosophy and Thoughtfulness",
        "Other"
    ]
    
    # Index Section
    summary_md += "#### 📌 Quick-Scan Index\n"
    summary_html += "<h4>📌 Quick-Scan Index</h4>"
    
    available_cats = sorted(grouped_briefs.keys(), key=lambda x: cat_order.index(x) if x in cat_order else 99)
    
    for cat in available_cats:
        summary_md += f"\n<span style='color:#1c83e1; font-size: 1.2em;'><b>{cat}</b></span>\n"
        summary_html += f"<p style='margin-bottom: 2px;'><span style='color:#1c83e1; font-size: 1.2em;'><b>{cat}</b></span></p><ul>"
        for v, _ in grouped_briefs[cat]:
            b = v.get('brief', {})
            summary_md += f"- **{b.get('channel')}**: [{b.get('episode_title')}]({v.get('url')}) *({b.get('duration_minutes')}m)*\n"
            summary_md += f"  > {b.get('one_line_summary')}\n"
            
            summary_html += f"<li><strong>{b.get('channel')}</strong>: <a href='{v.get('url')}'>{b.get('episode_title')}</a> <em>({b.get('duration_minutes')}m)</em><br/>"
            summary_html += f"<span style='font-size: 0.9em; color: #555;'>{b.get('one_line_summary')}</span></li>"
        summary_html += "</ul>"
        
    summary_md += "\n---\n\n"
    summary_html += "<hr/>"

    # Detailed Briefs Section
    final_md = summary_md
    final_html = summary_html
    
    for cat in available_cats:
        final_md += f"# 📁 Sector: {cat}\n\n"
        final_html += f"<h1 style='background: #f4f4f4; padding: 10px; border-left: 5px solid #00d1b2;'>Sector: {cat}</h1>"
        for _, (md, html) in grouped_briefs[cat]:
            final_md += md
            final_html += html
        
    final_html += "</body></html>"
    
    with open(md_filename, "w") as f:
        f.write(final_md)
        
    # Save JSON briefs for the dashboard to enable advanced visualizations
    json_filename = os.path.join("briefs", f"{date_str}.json")
    
    enriched_briefs = []
    for v in processed_queue:
        if v.get('brief'):
            b = v.get('brief')
            # Inject metadata for dashboard parity
            b['thumbnail'] = v.get('thumbnail', '')
            b['video_url'] = v.get('url', '')
            enriched_briefs.append(b)

    with open(json_filename, "w") as f:
        json.dump(enriched_briefs, f, indent=2)
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
