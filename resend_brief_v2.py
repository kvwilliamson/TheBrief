import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pipeline.summarization import format_html

# Setup minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resend_full_html_brief():
    load_dotenv()
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    json_path = f"briefs/{date_str}.json"
    
    if not os.path.exists(json_path):
        logger.error(f"Error: JSON Brief for {date_str} not found at {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Check if we have the new high-fidelity format
    if isinstance(data, dict) and "meta_summary" in data:
        meta_summary = data["meta_summary"]
        briefs = data["briefs"]
        clusters = data.get("clusters", [])
    else:
        # Legacy format fallback
        meta_summary = "Meta-summary unavailable for legacy brief."
        briefs = data if isinstance(data, list) else []
        clusters = []

    total_videos = len(briefs)
    total_time = sum(b.get('duration_minutes', 0) for b in briefs)

    # Build Header
    html = f"<html><body style='font-family: Arial, Helvetica, sans-serif; color: #202124; background-color: #ffffff; padding: 20px;'>"
    html += f"<h1 style='color: #202124; font-size: 24px; margin-bottom: 5px;'>TheBrief Daily Dispatch (RE-SEND)</h1>"
    html += f"<div style='color: #70757A; margin-bottom: 25px; font-size: 14px;'>Intelligence Report: {date_str}</div>"
    
    # Executive Meta Summary
    html += f"<div style='background: #1A73E8; color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px;'>"
    html += f"<h2 style='margin-top: 0; font-size: 20px;'>🧠 Executive Meta Summary</h2>"
    html += f"<div style='font-size: 14px; line-height: 1.6;'>{meta_summary.replace(chr(10), '<br/>')}</div></div>"

    # Narrative Clusters Index
    html += f"<h2 style='color: #202124; font-size: 20px; border-bottom: 2px solid #1A73E8; padding-bottom: 10px; margin-bottom: 20px; font-weight: bold;'>📁 Narrative Clusters</h2>"
    
    for cluster in clusters:
        header_bg = "#fde8e8" if cluster.get('is_crowded') else "#f8f9fa"
        html += f"<div style='margin-bottom: 25px; border: 1px solid #eee; border-radius: 8px; overflow: hidden;'>"
        html += f"<div style='background: {header_bg}; padding: 12px; border-bottom: 1px solid #eee;'>"
        html += f"<span style='font-weight: bold; font-size: 15px;'>Cluster {cluster['id']}: {' / '.join(cluster['themes'])}</span>"
        if cluster.get('is_crowded'):
            html += " <span style='background: #EA4335; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 10px;'>CROWDED TRADE</span>"
        html += f"<div style='font-size: 12px; color: #70757A; margin-top: 4px;'>{cluster['size']} channels • Avg Signal: {cluster['avg_signal']:.1f}/10</div></div>"
        
        html += "<table width='100%' style='border-collapse: collapse;'>"
        for b in briefs:
            if b.get('cluster_id') == cluster['id']:
                html += "<tr><td style='padding: 10px; border-bottom: 1px solid #f1f3f4;'>"
                html += f"<div style='font-size: 14px;'><b style='color: #202124;'>{b.get('channel')}:</b> <a href='{b.get('video_url')}' style='color: #1155CC; text-decoration: none;'>{b.get('episode_title')}</a></div>"
                html += f"<div style='font-size: 13px; color: #3C4043; margin-top: 4px;'>{b.get('one_line_summary')}</div>"
                html += "</td></tr>"
        html += "</table></div>"
        
    html += "<hr style='border: 0; border-top: 1px solid #eee; margin: 40px 0;' />"

    # Detailed Briefs
    for b in briefs:
        html += format_html(b, b.get('video_url', ''), b.get('thumbnail', ''))
    
    html += "</body></html>"

    # Send
    email_to = os.getenv("EMAIL_TO")
    email_from = os.getenv("EMAIL_FROM")
    smtp_host = os.getenv("SMTP_HOST")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    if not all([email_to, email_from, smtp_host, smtp_password]):
        logger.error("Missing email configuration.")
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
        
        recipients = [e.strip() for e in email_to.split(",")]
        for recipient in recipients:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"TheBrief Daily Digest (Resend) - {date_str}"
            msg["From"] = email_from
            msg["To"] = recipient
            msg.attach(MIMEText(html, "html"))
            server.sendmail(email_from, recipient, msg.as_string())
            logger.info(f"Full HTML Brief re-sent successfully to {recipient}.")
            
        server.quit()
    except Exception as e:
        logger.error(f"Failed to send: {e}")

if __name__ == "__main__":
    resend_full_html_brief()
