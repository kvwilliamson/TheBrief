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

# Define the expected Pydantic schema
class BriefSchema(BaseModel):
    episode_title: str = Field(description="Title of the episode")
    channel: str = Field(description="Name of the channel")
    duration_minutes: int = Field(description="Duration in minutes")
    tldr: str = Field(description="One sentence. Max 25 words.")
    key_takeaways: List[str] = Field(description="3-5 items")
    controversial_ideas: List[str] = Field(description="0-3 items")
    notable_quotes: List[str] = Field(description="1-2 verbatim quotes")
    topics_covered: List[str] = Field(description="comma-style tags")

def get_llm():
    model_choice = os.getenv("SUMMARY_MODEL", "gemini").lower()
    
    if model_choice == "gemini":
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.2,
                google_api_key=os.getenv("GOOGLE_AI_API_KEY")
            )
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}. Falling back to OpenAI.")
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
    print(f"Summarizing: {video['title']}")
    
    parser = JsonOutputParser(pydantic_object=BriefSchema)
    
    prompt = PromptTemplate(
        template="You are an expert transcriber and summarizer. Please analyze the following transcript and output a JSON object matching the exact format instructions.\n\n"
                 "Video details:\nTitle: {title}\nChannel: {channel}\nDuration: {duration_minutes} minutes\n\n"
                 "Transcript: {transcript}\n\nFormat instructions: {format_instructions}\n\n",
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
        print(f"Failed to parse LLM output for {video['title']}: {e}")
        return None
    except Exception as e:
        print(f"Error during summarization for {video['title']}: {e}")
        return None

def format_markdown(brief_json):
    md = f"## {brief_json['episode_title']} "
    md += f"*(Channel: {brief_json['channel']} | Length: {brief_json['duration_minutes']} min)*\n\n"
    md += f"**TL;DR:** {brief_json['tldr']}\n\n"
    
    md += "### Key Takeaways\n"
    for item in brief_json['key_takeaways']:
        md += f"- {item}\n"
        
    if brief_json.get('controversial_ideas'):
        md += "\n### Controversial Ideas\n"
        for item in brief_json['controversial_ideas']:
            md += f"- {item}\n"
            
    if brief_json.get('notable_quotes'):
        md += "\n### Notable Quotes\n"
        for item in brief_json['notable_quotes']:
            md += f"> \"{item}\"\n"
            
    md += "\n### Topics Covered\n"
    md += ", ".join(brief_json['topics_covered']) + "\n\n---\n\n"
    
    return md

def format_html(brief_json):
    html = f"<h2>{brief_json['episode_title']}</h2>"
    html += f"<p><em>(Channel: {brief_json['channel']} | Length: {brief_json['duration_minutes']} min)</em></p>"
    html += f"<p><strong>TL;DR:</strong> {brief_json['tldr']}</p>"
    
    html += "<h3>Key Takeaways</h3><ul>"
    for item in brief_json['key_takeaways']:
        html += f"<li>{item}</li>"
    html += "</ul>"
        
    if brief_json.get('controversial_ideas'):
        html += "<h3>Controversial Ideas</h3><ul>"
        for item in brief_json['controversial_ideas']:
            html += f"<li>{item}</li>"
        html += "</ul>"
            
    if brief_json.get('notable_quotes'):
        html += "<h3>Notable Quotes</h3><ul>"
        for item in brief_json['notable_quotes']:
            html += f"<li><em>\"{item}\"</em></li>"
        html += "</ul>"
            
    html += "<h3>Topics Covered</h3>"
    html += "<p>" + ", ".join(brief_json['topics_covered']) + "</p><hr/>"
    
    return html

def send_email_digest(html_content, date_str):
    print("Preparing email digest...")
    email_to = os.getenv("EMAIL_TO")
    email_from = os.getenv("EMAIL_FROM")
    smtp_host = os.getenv("SMTP_HOST")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    if not all([email_to, email_from, smtp_host, smtp_password]):
        print("Missing email configuration, skipping email delivery.")
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
            
        print(f"Connecting to SMTP server {host}:{port}")
        server = smtplib.SMTP(host, port)
        server.starttls()
        smtp_user = os.getenv("SMTP_USER", email_from)
        server.login(smtp_user, smtp_password)
        server.sendmail(email_from, email_to, msg.as_string())
        server.quit()
        print("Email digest sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

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
