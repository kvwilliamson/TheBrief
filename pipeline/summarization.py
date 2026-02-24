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
from typing import List
import google.generativeai as genai
import time

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
    evidence_strength: str = Field(description="Strictly one of: Low, Moderate, High")

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
    executive_use_case: ExecutiveUseCase
    core_claims: List[Claim] = Field(description="Combined core claims and forward projections (Max 6)")
    mechanism: Mechanism = Field(description="Plain-language mechanism summary")
    disconfirming_signals: List[str] = Field(description="Max 3 observable, time-bound disconfirming signals to watch")
    historical_parallel: str = Field(description="Optional one brief comparison to a historical parallel. Empty string if not applicable.")
    positioning_risk: str = Field(description="Strictly one of: Crowded, Neutral, Underowned, Unknown. Only if financial topic, otherwise empty.")
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
    logger.info(f"Summarizing natively via Audio: {video['title']}")
    
    parser = JsonOutputParser(pydantic_object=BriefSchema)
    
    # 1. Upload Audio to Gemini
    audio_path = video.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Missing audio path for {video['title']}")
        return None
        
    genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
    logger.info(f"  Uploading {audio_path} to Gemini...")
    file_handle = genai.upload_file(path=audio_path, mime_type="audio/mpeg")
    
    # Wait for processing
    while file_handle.state.name == "PROCESSING":
        time.sleep(2)
        file_handle = genai.get_file(file_handle.name)
        
    if file_handle.state.name == "FAILED":
        logger.error(f"Gemini File API processing failed for {video['title']}")
        return None
        
    logger.info(f"  File ready. Generating High-Utility Decision Brief...")
    
    # 2. Build the LLM Chain natively passing the file_handle
    prompt = PromptTemplate(
        template="You are an expert intelligence analyst generating a high-utility, decision-oriented Intelligence Brief directly from an audio file.\n"
                 "Listen to the attached audio and extract the core thesis and arguments.\n\n"
                 "Your strict constraints:\n"
                 "1. Tone: Intelligence memo, not analyst blog. No academic verbosity, no marketing tone, no emotional language, no hype formatting, no emojis.\n"
                 "2. Limit bullets per section to 6. Max total word count 400-600 words.\n"
                 "3. Short paragraphs. No section may exceed 25% of total length. If thesis is repetitive, shorten proportionally.\n"
                 "4. Output MUST conform exactly to the JSON schema.\n"
                 "5. Disconfirming signals must be observable, time-bound, and max 3 items. No generic macro hedging language.\n"
                 "6. Provide a Historical Parallel if applicable in one tight paragraph.\n\n"
                 "Video details:\nTitle: {title}\nChannel: {channel}\nDuration: {duration_minutes} minutes\n\n"
                 "Format instructions:\n{format_instructions}\n\n",
        input_variables=["title", "channel", "duration_minutes"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    try:
        # Langchain Google GenAI accepts a list of [prompt_text, gemini_file_uri]
        formatted_prompt = prompt.format(
            title=video.get("title", ""),
            channel=video.get("channel", ""),
            duration_minutes=video.get("duration_minutes", 0)
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
            genai.delete_file(file_handle.name)
        except Exception as e:
            logger.warning(f"Failed to delete file from Gemini API: {e}")
            
        # Delete local copy
        try:
            os.remove(audio_path)
        except OSError:
            pass

def format_markdown(brief: dict) -> str:
    md = f"## {brief.get('episode_title', 'Unknown Title')}\n"
    md += f"**{brief.get('channel', 'Unknown')}** | **Length:** {brief.get('duration_minutes', 0)} min | **Domain:** {brief.get('topic_domain', 'Unknown')}\n\n"
    
    euc = brief.get('executive_use_case', {})
    md += "### Executive Use Case\n"
    md += f"- **Signal type:** {euc.get('signal_type')}\n"
    md += f"- **Positioning impact:** {euc.get('positioning_impact')}\n"
    md += f"- **Time horizon:** {euc.get('time_horizon')}\n"
    md += f"- **Confidence:** {euc.get('confidence_level')}\n"
    md += f"- **Incentive bias:** {euc.get('incentive_bias')}\n"
    md += f"- **Consensus context:** {euc.get('consensus_context')}\n"
    
    pos_risk = brief.get('positioning_risk')
    if pos_risk:
        md += f"- **Positioning Risk:** {pos_risk}\n"
    md += "\n"

    claims = brief.get('core_claims', [])
    if claims:
        md += "### Core Claims\n"
        for i, claim in enumerate(claims[:6], 1):
            md += f"**Claim {i}:** {claim.get('claim')}\n"
            md += f"- **Evidence:** {claim.get('evidence_cited')} (Type: {claim.get('evidence_type')})\n"
            md += f"- **Strength:** {claim.get('evidence_strength')}\n\n"

    mech = brief.get('mechanism', {})
    if mech:
        md += "### Mechanism (If True)\n"
        md += f"- **Trigger:** {mech.get('trigger')}\n"
        md += f"- **Transmission:** {mech.get('transmission_path')}\n"
        md += f"- **Impact:** {mech.get('market_impact')}\n"
        md += f"- **Secondary effects:** {mech.get('secondary_effects')}\n\n"

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

def format_html(brief: dict) -> str:
    html = f"<h2>{brief.get('episode_title', 'Unknown Title')}</h2>"
    html += f"<p><strong>{brief.get('channel', 'Unknown')}</strong> | <strong>Length:</strong> {brief.get('duration_minutes', 0)} min | <strong>Domain:</strong> {brief.get('topic_domain', 'Unknown')}</p>"
    
    euc = brief.get('executive_use_case', {})
    html += "<h3>Executive Use Case</h3><ul>"
    html += f"<li><strong>Signal type:</strong> {euc.get('signal_type')}</li>"
    html += f"<li><strong>Positioning impact:</strong> {euc.get('positioning_impact')}</li>"
    html += f"<li><strong>Time horizon:</strong> {euc.get('time_horizon')}</li>"
    html += f"<li><strong>Confidence:</strong> {euc.get('confidence_level')}</li>"
    html += f"<li><strong>Incentive bias:</strong> {euc.get('incentive_bias')}</li>"
    html += f"<li><strong>Consensus context:</strong> {euc.get('consensus_context')}</li>"
    
    pos_risk = brief.get('positioning_risk')
    if pos_risk:
        html += f"<li><strong>Positioning risk:</strong> {pos_risk}</li>"
    html += "</ul>"

    claims = brief.get('core_claims', [])
    if claims:
        html += "<h3>Core Claims</h3>"
        for i, claim in enumerate(claims[:6], 1):
            html += f"<p><strong>Claim {i}:</strong> {claim.get('claim')}<br/>"
            html += f"&nbsp;&nbsp;&nbsp;&nbsp;<strong>Evidence:</strong> {claim.get('evidence_cited')} (Type: {claim.get('evidence_type')})<br/>"
            html += f"&nbsp;&nbsp;&nbsp;&nbsp;<strong>Strength:</strong> {claim.get('evidence_strength')}</p>"

    mech = brief.get('mechanism', {})
    if mech:
        html += "<h3>Mechanism (If True)</h3><ul>"
        html += f"<li><strong>Trigger:</strong> {mech.get('trigger')}</li>"
        html += f"<li><strong>Transmission:</strong> {mech.get('transmission_path')}</li>"
        html += f"<li><strong>Impact:</strong> {mech.get('market_impact')}</li>"
        html += f"<li><strong>Secondary effects:</strong> {mech.get('secondary_effects')}</li></ul>"

    signals = brief.get('disconfirming_signals', [])
    if signals:
        html += "<h3>Disconfirming Signals to Watch</h3><ul>"
        for sig in signals[:3]:
            html += f"<li>{sig}</li>"
        html += "</ul>"

    hist = brief.get('historical_parallel')
    if hist:
        html += "<h3>Historical Parallel</h3>"
        html += f"<p>{hist}</p>"

    html += "<hr/>"
    return html

def send_email_digest(html_content, date_str):
    logger.info("Preparing email digest...")
    email_to = os.getenv("EMAIL_TO")
    email_from = os.getenv("EMAIL_FROM")
    smtp_host = os.getenv("SMTP_HOST")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    if not all([email_to, email_from, smtp_host, smtp_password]):
        logger.info("Missing email configuration, skipping email delivery.")
        return
        
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"TheBrief Daily Digest - {date_str}"
    msg["From"] = email_from
    msg["To"] = email_to
    
    part = MIMEText(html_content, "html")
    msg.attach(part)
    
    try:
        host = smtp_host
        port = 587
        if ":" in smtp_host:
            host, port_str = smtp_host.split(":")
            port = int(port_str)
            
        logger.info(f"Connecting to SMTP server {host}:{port}")
        server = smtplib.SMTP(host, port)
        server.starttls()
        smtp_user = os.getenv("SMTP_USER", email_from)
        server.login(smtp_user, smtp_password)
        server.sendmail(email_from, email_to, msg.as_string())
        server.quit()
        logger.info("Email digest sent successfully.")
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

    def process_video_summary(video):
        # Transcripts no longer required; we pass audio directly
        brief = summarize_transcript(video, llm)
        if brief:
            video["brief"] = brief
            # Generate markdown and HTML components
            md = format_markdown(brief)
            html = format_html(brief)
            
            # Update DB to mark as processed
            db_path = os.path.join("data", "processed_videos.json")
            try:
                from tinydb import TinyDB
                db = TinyDB(db_path)
                db.insert({"id": video["id"], "title": video["title"], "processed_at": datetime.now().isoformat()})
            except Exception as e:
                logger.warning(f"Note: Error updating processed_videos db: {e}")
                
            return video, (md, html)
        return None, None

    # Run summarization concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_video = {executor.submit(process_video_summary, video): video for video in queue}
        for future in concurrent.futures.as_completed(future_to_video):
            v_res, format_res = future.result()
            if v_res and format_res:
                processed_queue.append(v_res)
                briefs_content.append(format_res)
                
    if not processed_queue:
        logger.warning("No briefs were successfully generated.")
        return []
        
    # Compile final outputs
    date_str = datetime.now().strftime("%Y-%m-%d")
    md_filename = os.path.join("briefs", f"{date_str}.md")
    os.makedirs("briefs", exist_ok=True)
    
    # Sort briefs by original queue order (or just append)
    final_md = ""
    final_html = f"<html><body><h1>TheBrief Daily Digest - {date_str}</h1>"
    
    for md, html in briefs_content:
        final_md += md
        final_html += html
        
    final_html += "</body></html>"
    
    # Write or append to the daily Markdown file
    mode = "a" if os.path.exists(md_filename) else "w"
    with open(md_filename, mode) as f:
        f.write(final_md)
        
    logger.info(f"Summarization complete. {len(processed_queue)} briefs written to {md_filename}")
    
    send_email = str(os.getenv("SEND_EMAIL", "false")).lower() == "true"
        
    # Clear queue after successful processing
    with open(queue_path, "w") as f:
        json.dump([], f)
    logger.info("Queue cleared.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_summarization()
