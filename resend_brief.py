import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from datetime import datetime

def resend_brief():
    load_dotenv()
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    md_path = f"briefs/{date_str}.md"
    
    if not os.path.exists(md_path):
        print(f"Error: Brief for {date_str} not found.")
        return

    with open(md_path, "r") as f:
        md_content = f.read()

    # Create a simple HTML version from the MD (best effort)
    # We'll just wrap it in a div with some styles to make it look decent
    html_content = f"""
    <html>
    <body style="font-family: sans-serif; background-color: #121212; color: #ffffff; padding: 20px;">
        <div style="max-width: 800px; margin: 0 auto; line-height: 1.6;">
            <h1 style="color: #1c83e1; border-bottom: 1px solid #333; padding-bottom: 10px;">TheBrief Daily Digest - {date_str}</h1>
            <div style="white-space: pre-wrap; font-family: 'Courier New', Courier, monospace; background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333;">
{md_content}
            </div>
            <p style="color: #888; font-size: 0.8em; margin-top: 20px; border-top: 1px solid #333; padding-top: 10px;">
                This is a re-sent copy of today's intelligence brief.
            </p>
        </div>
    </body>
    </html>
    """

    email_to = "kochava72@gmail.com" # Specifically send to the person who missed it
    email_from = os.getenv("EMAIL_FROM")
    smtp_host = os.getenv("SMTP_HOST")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    if not all([email_to, email_from, smtp_host, smtp_password]):
        print("Missing config.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"RESEND: TheBrief Daily Digest - {date_str}"
    msg["From"] = email_from
    msg["To"] = email_to
    
    msg.attach(MIMEText(html_content, "html"))
    
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
        server.sendmail(email_from, email_to, msg.as_string())
        server.quit()
        print(f"Brief successfully resent to {email_to}")
    except Exception as e:
        print(f"Failed to resend: {e}")

if __name__ == "__main__":
    resend_brief()
