import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv

# Global Logging Setup
os.makedirs("data", exist_ok=True)
log_file = os.path.join("data", "pipeline.log")

# Silence verbose library logging
logging.getLogger("googleapiclient").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2),
        logging.StreamHandler(sys.stdout)
    ]
)

# Imports must happen AFTER root logger is configured
from pipeline.discovery import run_discovery
from pipeline.extraction import run_extraction
from pipeline.summarization import run_summarization

def main():
    logging.info("="*50)
    logging.info("Starting TheBrief Daily Pipeline")
    logging.info("="*50)
    
    # Load environment variables (Local only)
    if not os.getenv("GITHUB_ACTIONS"):
        load_dotenv()
    
    # Verify core API keys
    yt_key = os.getenv("YOUTUBE_API_KEY")
    if not yt_key:
        logging.error("CRITICAL: YOUTUBE_API_KEY is missing or empty!")
        sys.exit(1)
        
    google_key = os.getenv("GOOGLE_AI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not google_key and not openai_key:
        logging.error("CRITICAL: Neither GOOGLE_AI_API_KEY nor OPENAI_API_KEY found!")
        sys.exit(1)

    # Phase 1
    logging.info("\n--- Phase 1: Discovery ---")
    run_discovery()
    
    # Phase 2
    logging.info("\n--- Phase 2: Extraction ---")
    run_extraction()
    
    # Phase 3
    logging.info("\n--- Phase 3: Summarization (Direct Audio) ---")
    run_summarization()
    
    logging.info("="*50)
    logging.info("Pipeline Execution Complete")
    logging.info("="*50)

if __name__ == "__main__":
    main()
