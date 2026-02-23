import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

from pipeline.discovery import run_discovery
from pipeline.extraction import run_extraction
from pipeline.transcription import run_transcription
from pipeline.summarization import run_summarization

# Global Logging Setup
os.makedirs("data", exist_ok=True)
log_file = os.path.join("data", "pipeline.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    logging.info("="*50)
    logging.info("Starting TheBrief Daily Pipeline")
    logging.info("="*50)
    
    # Load environment variables
    load_dotenv()
    
    # Verify core API keys
    if not os.getenv("YOUTUBE_API_KEY"):
        logging.error("CRITICAL: YOUTUBE_API_KEY is missing!")
        return
        
    if not os.getenv("GOOGLE_AI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        logging.error("CRITICAL: Neither GOOGLE_AI_API_KEY nor OPENAI_API_KEY found!")
        return

    # Phase 1
    logging.info("\n--- Phase 1: Discovery ---")
    run_discovery()
    
    # Phase 2
    logging.info("\n--- Phase 2: Extraction ---")
    run_extraction()
    
    # Phase 3
    logging.info("\n--- Phase 3: Transcription ---")
    run_transcription()
    
    # Phase 4
    logging.info("\n--- Phase 4: Summarization ---")
    run_summarization()
    
    logging.info("\n="*50)
    logging.info("Pipeline Execution Complete")
    logging.info("="*50)

if __name__ == "__main__":
    main()
