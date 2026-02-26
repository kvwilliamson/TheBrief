import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pipeline.summarization import format_html, get_profile_for_category

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
        briefs = json.load(f)

    # Re-construct the HTML exactly as the pipeline does
    cat_order = [
        "General Financial Investing and Speculation",
        "Precious Metals",
        "Artificial Intelligence",
        "Health and Nutrition",
        "Philosophy and Thoughtfulness",
        "Other"
    ]
    
    # Sort into categories
    from collections import defaultdict
    grouped = defaultdict(list)
    for b in briefs:
        # Note: the JSON stores the brief object, we need to map back to category
        # In summarization.py we injected thumbnail and video_url into the brief object
        cat = b.get('topic_domain', 'Other')
        grouped[cat].append(b)

    total_videos = len(briefs)
    total_time = sum(b.get('duration_minutes', 0) for b in briefs)

    # Build Header
    html = f"<html><body style='font-family: Arial, Helvetica, sans-serif; color: #202124; background-color: #ffffff; padding: 20px;'>"
    html += f"<h1 style='color: #202124; font-size: 24px; margin-bottom: 5px;'>TheBrief Daily Dispatch (RE-SEND)</h1>"
    html += f"<div style='color: #70757A; margin-bottom: 25px; font-size: 14px;'>Intelligence Report: {date_str}</div>"
    
    html += f"<div style='background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 30px;'>"
    html += f"<h3 style='margin-top: 0; color: #202124; font-size: 18px;'>📊 At a Glance</h3>"
    html += f"<ul style='margin-bottom: 0; font-size: 14px;'>"
    html += f"<li><strong>Total Intelligence Assets:</strong> {total_videos} videos</li>"
    html += f"<li><strong>Total Subject Time:</strong> {total_time:.1f} minutes</li></ul></div>"

    # --- QUICK SCAN INDEX (Grouped + Small Thumbnails) ---
    html += f"<h2 style='color: #202124; font-size: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 20px; font-weight: bold;'>📌 Quick-Scan Index</h2>"
    
    available_cats = sorted(grouped.keys(), key=lambda x: cat_order.index(x) if x in cat_order else 99)

    for cat in available_cats:
        html += f"<div style='margin-top: 25px; margin-bottom: 15px; color: #1A73E8; font-size: 16px; font-weight: bold;'>{cat}</div>"
        html += "<table width='100%' style='border-collapse: collapse;'>"
        for b in grouped[cat]:
            html += "<tr>"
            html += f"<td style='padding: 10px 10px 10px 0; vertical-align: top;'>"
            html += f"<div style='font-size: 14px; margin-bottom: 4px;'>"
            html += f"<b style='color: #202124;'>{b.get('channel')}:</b> <a href='{b.get('video_url')}' style='color: #1155CC; text-decoration: none;'>{b.get('episode_title')}</a> "
            html += f"<span style='color: #70757A; font-style: italic;'>({b.get('duration_minutes')}m)</span></div>"
            html += f"<div style='font-size: 13.5px; color: #3C4043; line-height: 1.4;'>{b.get('one_line_summary')}</div>"
            html += "</td></tr>"
        html += "</table>"
        
    html += "<hr style='border: 0; border-top: 1px solid #eee; margin: 40px 0;' />"

    # --- DETAILED BRIEFS SECTION (Grouped) ---
    for cat in available_cats:
        html += f"<h1 style='background: #1A73E8; color: white; padding: 12px; border-radius: 4px; font-size: 20px; margin-bottom: 30px; font-weight: bold;'>Sector: {cat}</h1>"
        for b in grouped[cat]:
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
            msg["Subject"] = f"TheBrief Daily Digest (Optimized) - {date_str}"
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
