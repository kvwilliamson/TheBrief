import os
import json
import logging
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

logger = logging.getLogger(__name__)

# --- Intelligence Brief Schemas (BKM v2) ---

class SignalSnapshot(BaseModel):
    evidence_strength: str = Field(description="Score out of 10 with brief qualifier (e.g. 7/10 - Strong empirical data)")
    analytical_depth: str = Field(description="Score out of 10 with brief qualifier")
    narrative_pressure: str = Field(description="Score out of 10 with brief qualifier")
    speculation_load: str = Field(description="Percentage estimate (e.g., 40%)")
    novelty_level: str = Field(description="Low / Medium / High")
    incentive_alignment: str = Field(description="Low / Moderate / High")

class HardClaim(BaseModel):
    claim: str = Field(description="High-impact, verifiable claim")

class RealityLayer(BaseModel):
    hard_claims: List[HardClaim] = Field(description="Only high-impact verifiable claims")
    overall_evidence_pattern: str = Field(description="1-2 line summary of evidence pattern")
    verification_density: str = Field(description="1-2 line summary of verification density")

class ForwardProjection(BaseModel):
    prediction: str = Field(description="State prediction")
    required_assumptions: List[str] = Field(description="Assumptions required for prediction")
    base_rate_context: str = Field(description="Base-rate context if available, else 'N/A'")
    falsifiability_condition: str = Field(description="Brief falsifiability condition")

class CausalNode(BaseModel):
    cause: str = Field(description="Cause")
    mechanism: str = Field(description="Mechanism")
    outcome: str = Field(description="Outcome")
    mechanism_clarity_score: str = Field(description="Score out of 10")

class NarrativeSubscores(BaseModel):
    emotional_amplification: int = Field(description="0-10 intensity")
    certainty_overreach: int = Field(description="0-10 intensity")
    catastrophic_framing: int = Field(description="0-10 intensity")
    tribal_framing: int = Field(description="0-10 intensity")
    institutional_distrust: int = Field(description="0-10 intensity")
    sensationalism: int = Field(description="0-10 intensity")

class NarrativeProfile(BaseModel):
    subscores: NarrativeSubscores
    summary: str = Field(description="2-3 line summary of narrative profile")

class IncentiveVector(BaseModel):
    monetization_model: str = Field(description="Monetization model")
    alignment_strength: str = Field(description="Alignment strength")
    transparency_level: str = Field(description="Transparency level")

class SignalToNarrativeRatio(BaseModel):
    signal_pct: int = Field(description="Signal (data + mechanism) percentage")
    narrative_pct: int = Field(description="Narrative (emotion + repetition) percentage")
    novel_information: str = Field(description="Low / Medium / High")

class FinalIntelligenceTake(BaseModel):
    classification: str = Field(description="Actionable / Sentiment Indicator / Background Context / Noise")
    strategic_assessment: str = Field(description="Max 5 sentences short strategic assessment")

class BriefSchema(BaseModel):
    episode_title: str = Field(description="Title of the episode")
    channel: str = Field(description="Name of the channel")
    duration_minutes: int = Field(description="Duration in minutes")
    topic_domain: str = Field(description="Broad topic domain")
    central_thesis: str = Field(description="Strict 1 sentence compression of the episode's core claim")
    signal_snapshot: SignalSnapshot
    reality_layer: RealityLayer
    forward_projections: List[ForwardProjection]
    causal_map: List[CausalNode]
    narrative_profile: NarrativeProfile
    incentive_vector: IncentiveVector
    signal_to_narrative_ratio: SignalToNarrativeRatio
    disconfirming_conditions: List[str] = Field(description="3-5 concrete disconfirming conditions that break the thesis")
    final_intelligence_take: FinalIntelligenceTake
def get_llm():
    model_choice = os.getenv("SUMMARY_MODEL", "gemini").lower()
    
    if model_choice == "gemini":
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
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
    logger.info(f"Summarizing: {video['title']}")
    
    parser = JsonOutputParser(pydantic_object=BriefSchema)
    
    prompt = PromptTemplate(
        template="You are an expert epistemic intelligence analyst generating a high-signal, decision-grade Intelligence Brief. "
                 "Your strict constraints:\n"
                 "1. Tone: analytical, sharp, compressed. Forbid academic verbosity, forbid marketing tone, forbid emotional language.\n"
                 "2. Limit bullets per section to 6. Enforce 400-700 word output total.\n"
                 "3. Enforce exactly one sentence for central thesis.\n"
                 "4. Enforce 5-sentence max for Final Intelligence Take.\n"
                 "5. Prioritize signal density over completeness.\n"
                 "6. Do not restate narrative language. Classify it.\n"
                 "7. Optimize for decision usefulness, not summarization completeness.\n\n"
                 "Analyze the following transcript and output a JSON object matching the exact format instructions.\n\n"
                 "Video details:\nTitle: {title}\nChannel: {channel}\nDuration: {duration_minutes} minutes\n\n"
                 "Transcript:\n{transcript}\n\n"
                 "Format instructions:\n{format_instructions}\n\n",
        input_variables=["title", "channel", "duration_minutes", "transcript"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "title": video["title"],
            "channel": video["channel"],
            "duration_minutes": int(video.get("duration_minutes", 0)),
            "transcript": video.get("transcript", "")
        })
        return result
    except OutputParserException as e:
        logger.error(f"Failed to parse LLM output for {video['title']}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during summarization for {video['title']}: {e}")
        return None

def format_markdown(brief: dict) -> str:
    md = f"## {brief.get('episode_title', 'Unknown Title')} "
    md += f"*(Channel: {brief.get('channel', 'Unknown')} | Length: {brief.get('duration_minutes', 0)} min)*\n\n"
    md += f"**Topic Domain:** {brief.get('topic_domain', 'Unknown')}\n"
    md += f"**Central Thesis:** {brief.get('central_thesis', 'Unknown')}\n\n"


    real = brief.get('reality_layer', {})
    md += "### 📊 WHAT WAS SAID (Reality Layer)\n"
    for claim in real.get('hard_claims', []):
        md += f"- {claim.get('claim')}\n"
    md += f"\n**Overall evidence pattern:** {real.get('overall_evidence_pattern')}\n"
    md += f"**Verification density:** {real.get('verification_density')}\n\n"

    projs = brief.get('forward_projections', [])
    if projs:
        md += "### 🔮 FORWARD PROJECTIONS (Risk Layer)\n"
        for p in projs[:6]:
            md += f"- **Prediction:** {p.get('prediction')}\n"
            md += f"  - *Assumptions:* {', '.join(p.get('required_assumptions', []))}\n"
            md += f"  - *Base-rate context:* {p.get('base_rate_context')}\n"
            md += f"  - *Falsifiability:* {p.get('falsifiability_condition')}\n"
        md += "\n"

    caus = brief.get('causal_map', [])
    if caus:
        md += "### 🧠 CAUSAL MAP\n"
        md += "| Cause | Mechanism | Outcome | Clarity |\n|---|---|---|---|\n"
        for c in caus[:6]:
            md += f"| {c.get('cause')} | {c.get('mechanism')} | {c.get('outcome')} | {c.get('mechanism_clarity_score')} |\n"
        md += "\n"


    inc = brief.get('incentive_vector', {})
    md += "### 💰 INCENTIVE VECTOR\n"
    md += f"- **Monetization Model:** {inc.get('monetization_model')}\n"
    md += f"- **Alignment Strength:** {inc.get('alignment_strength')}\n"
    md += f"- **Transparency Level:** {inc.get('transparency_level')}\n\n"

    ratio = brief.get('signal_to_narrative_ratio', {})
    md += "### 📡 SIGNAL-TO-NARRATIVE RATIO\n"
    md += f"**Signal:** {ratio.get('signal_pct')}% | **Narrative:** {ratio.get('narrative_pct')}% | **Novel Information:** {ratio.get('novel_information')}\n\n"

    md += "### 🧨 WHAT WOULD BREAK THIS THESIS?\n"
    for cond in brief.get('disconfirming_conditions', [])[:5]:
        md += f"- {cond}\n"
    md += "\n"

    take = brief.get('final_intelligence_take', {})
    md += "### 🏁 FINAL INTELLIGENCE TAKE\n"
    md += f"**Classification:** {take.get('classification')}\n\n"
    md += f"{take.get('strategic_assessment')}\n\n"
    
    md += f"---\n\n"
    return md

def format_html(brief: dict) -> str:
    html = f"<h2>{brief.get('episode_title', 'Unknown Title')}</h2>"
    html += f"<p><em>(Channel: {brief.get('channel', 'Unknown')} | Length: {brief.get('duration_minutes', 0)} min)</em></p>"
    html += f"<p><strong>Topic Domain:</strong> {brief.get('topic_domain', 'Unknown')}<br/>"
    html += f"<strong>Central Thesis:</strong> {brief.get('central_thesis', 'Unknown')}</p>"


    real = brief.get('reality_layer', {})
    html += "<h3>📊 WHAT WAS SAID (Reality Layer)</h3><ul>"
    for claim in real.get('hard_claims', []):
        html += f"<li>{claim.get('claim')}</li>"
    html += f"</ul><p><strong>Overall evidence pattern:</strong> {real.get('overall_evidence_pattern')}<br/>"
    html += f"<strong>Verification density:</strong> {real.get('verification_density')}</p>"

    projs = brief.get('forward_projections', [])
    if projs:
        html += "<h3>🔮 FORWARD PROJECTIONS (Risk Layer)</h3><ul>"
        for p in projs[:6]:
            html += f"<li><strong>Prediction:</strong> {p.get('prediction')}<ul>"
            html += f"<li><em>Assumptions:</em> {', '.join(p.get('required_assumptions', []))}</li>"
            html += f"<li><em>Base-rate context:</em> {p.get('base_rate_context')}</li>"
            html += f"<li><em>Falsifiability:</em> {p.get('falsifiability_condition')}</li></ul></li>"
        html += "</ul>"

    caus = brief.get('causal_map', [])
    if caus:
        html += "<h3>🧠 CAUSAL MAP</h3>"
        html += "<table border='1'><tr><th>Cause</th><th>Mechanism</th><th>Outcome</th><th>Clarity</th></tr>"
        for c in caus[:6]:
            html += f"<tr><td>{c.get('cause')}</td><td>{c.get('mechanism')}</td><td>{c.get('outcome')}</td><td>{c.get('mechanism_clarity_score')}</td></tr>"
        html += "</table>"


    inc = brief.get('incentive_vector', {})
    html += "<h3>💰 INCENTIVE VECTOR</h3><ul>"
    html += f"<li><strong>Monetization Model:</strong> {inc.get('monetization_model')}</li>"
    html += f"<li><strong>Alignment Strength:</strong> {inc.get('alignment_strength')}</li>"
    html += f"<li><strong>Transparency Level:</strong> {inc.get('transparency_level')}</li></ul>"

    ratio = brief.get('signal_to_narrative_ratio', {})
    html += "<h3>📡 SIGNAL-TO-NARRATIVE RATIO</h3>"
    html += f"<p><strong>Signal:</strong> {ratio.get('signal_pct')}% | <strong>Narrative:</strong> {ratio.get('narrative_pct')}% | <strong>Novel Information:</strong> {ratio.get('novel_information')}</p>"

    html += "<h3>🧨 WHAT WOULD BREAK THIS THESIS?</h3><ul>"
    for cond in brief.get('disconfirming_conditions', [])[:5]:
        html += f"<li>{cond}</li>"
    html += "</ul>"

    take = brief.get('final_intelligence_take', {})
    html += "<h3>🏁 FINAL INTELLIGENCE TAKE</h3>"
    html += f"<p><strong>Classification:</strong> {take.get('classification')}</p>"
    html += f"<p>{take.get('strategic_assessment')}</p><hr/>"
    
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
        logger.error("No queue found. Run transcription first.")
        return
        
    with open(queue_path, "r") as f:
        queue = json.load(f)
        
    if not queue:
        logger.info("Queue is empty, nothing to summarize.")
        return
        
    llm = get_llm()
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("briefs", exist_ok=True)
    md_file_path = os.path.join("briefs", f"{date_str}.md")
    
    all_html_content = f"<h1>TheBrief - Daily Digest ({date_str})</h1>"
    
    briefs_generated = 0
    with open(md_file_path, "a") as md_file:
        for video in queue:
            if not video.get("transcript"):
                continue
                
            brief_json = summarize_transcript(video, llm)
            if brief_json:
                md_content = format_markdown(brief_json)
                md_file.write(md_content)
                all_html_content += format_html(brief_json)
                briefs_generated += 1
                
                # Update DB to mark as processed
                db_path = os.path.join("data", "processed_videos.json")
                try:
                    from tinydb import TinyDB
                    db = TinyDB(db_path)
                    db.insert({"id": video["id"], "title": video["title"], "processed_at": datetime.now().isoformat()})
                except Exception as e:
                    logger.warning(f"Note: Error updating processed_videos db: {e}")
                    
    logger.info(f"Summarization complete. {briefs_generated} briefs written to {md_file_path}")
    
    send_email = str(os.getenv("SEND_EMAIL", "false")).lower() == "true"
    if send_email and briefs_generated > 0:
        send_email_digest(all_html_content, date_str)
        
    # Clear queue after successful processing
    with open(queue_path, "w") as f:
        json.dump([], f)
    logger.info("Queue cleared.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_summarization()
