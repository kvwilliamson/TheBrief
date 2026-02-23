# TheBrief

TheBrief is an automated daily pipeline that monitors a curated list of YouTube podcast channels, extracts audio from new long-form episodes, transcribes them, and delivers a structured AI-generated briefing. The output is a clean, scannable summary that distills hours of audio into minutes of reading.

## System Architecture

The pipeline runs as a single Python script (`main.py`) with four discrete stages:

1. **Discovery**: Monitors configured channels in `channels.json` and identifies new episodes published in the last 24 hours that are longer than 20 minutes (excluding Shorts).
2. **Audio Extraction**: Extracts only the audio stream from each queued video using `yt-dlp` and downsamples it to 16kHz mono `mp3` to optimize for Whisper transcription. No video is downloaded.
3. **Transcription**: Converts the audio to a full text transcript. Can be configured to run locally (free) via `faster-whisper` or via the OpenAI Whisper API.
4. **Summarization & Delivery**: Feeds each transcript into an LLM (Google Gemini 1.5 Pro, with an OpenAI fallback) with a structured prompt to generate a JSON summary. The summary is saved as a Markdown file in `/briefs/` and optionally emailed.

## Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kvwilliamson/TheBrief.git
   cd TheBrief
   ```

2. **Set up a virtual environment and install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy the example environment file and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and configure:
   - `YOUTUBE_API_KEY`: Required for Phase 1 to find new videos.
   - `GOOGLE_AI_API_KEY`: Required for Gemini summarization.
   - `OPENAI_API_KEY`: Required for Whisper API transcription (and as a fallback for summarization).
   - Email Delivery (Optional): Set `SEND_EMAIL=true` and provide your `EMAIL_TO`, `EMAIL_FROM`, `SMTP_HOST` (e.g. `smtp.gmail.com:587`), and `SMTP_PASSWORD` (use an App Password, not your real password).

4. **Curate Channels**:
   Update `channels.json` with the names and actual YouTube Channel IDs you wish to track.

5. **Run the Dashboard UI (Recommended)**:
   ```bash
   streamlit run app.py
   ```
   *This will open a local web page where you can view your briefs, manage channels, and manually trigger the pipeline.*

6. **Run the Pipeline Headless**:
   ```bash
   python main.py
   ```

## GitHub Actions

The pipeline is pre-configured to run automatically every day at 6:00 AM UTC via GitHub Actions (`.github/workflows/daily_brief.yml`). 

To enable this action on GitHub:
1. Go to your repository **Settings** > **Secrets and variables** > **Actions**.
2. Add the following **Repository Secrets**:
   - `YOUTUBE_API_KEY`
   - `GOOGLE_AI_API_KEY`
   - `OPENAI_API_KEY`
   - (Optional) `SMTP_PASSWORD` and other email credentials if you update the workflow to pass them in. 
3. The cron job will run daily, process new podcasts, and commit the generated Markdown summaries directly back to the `briefs/` folder in the repository.
