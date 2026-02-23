import os
from dotenv import load_dotenv

from pipeline.discovery import run_discovery
from pipeline.extraction import run_extraction
from pipeline.transcription import run_transcription
from pipeline.summarization import run_summarization

def main():
    print("="*50)
    print("Starting TheBrief Daily Pipeline")
    print("="*50)
    
    # Load environment variables
    load_dotenv()
    
    # Verify core API keys
    if not os.getenv("YOUTUBE_API_KEY"):
        print("CRITICAL: YOUTUBE_API_KEY is missing!")
        return
        
    if not os.getenv("GOOGLE_AI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL: Neither GOOGLE_AI_API_KEY nor OPENAI_API_KEY found!")
        return

    # Phase 1
    print("\n--- Phase 1: Discovery ---")
    run_discovery()
    
    # Phase 2
    print("\n--- Phase 2: Extraction ---")
    run_extraction()
    
    # Phase 3
    print("\n--- Phase 3: Transcription ---")
    run_transcription()
    
    # Phase 4
    print("\n--- Phase 4: Summarization ---")
    run_summarization()
    
    print("\n="*50)
    print("Pipeline Execution Complete")
    print("="*50)

if __name__ == "__main__":
    main()
