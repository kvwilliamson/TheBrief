# TheBrief

TheBrief is an automated intelligence pipeline that monitors a curated list of long-form YouTube podcast channels, extracts audio from newly published episodes, transcribes them, and generates a structured **BKM v2 Intelligence Brief**.

The output is a high-signal, decision-grade dashboard that compresses hours of audio into under three minutes of disciplined, structured reading per episode.

> This is not a summary engine.
> It is an epistemic signal extractor.

## Core Philosophy

TheBrief enforces:
- Strict separation of fact vs speculation
- Explicit narrative pressure scoring
- Incentive alignment analysis
- Disconfirming conditions
- Signal-to-narrative ratio estimation
- Strategic classification (Actionable / Sentiment / Context / Noise)

> The goal is not completeness.
> The goal is decision usefulness.

## System Architecture

The pipeline runs as a single Python orchestrator (`main.py`) composed of three discrete stages:

### 1. Discovery
- Monitors configured channels in `channels.json`
- Identifies new episodes published in the last 24 hours
- Filters for long-form content (>20 minutes)
- Excludes Shorts automatically

### 2. Audio Extraction
- Uses `yt-dlp`
- Extracts audio stream only (no video download)
- Downsamples to 16kHz mono `mp3`
- Optimized for transcription efficiency

### 3. Intelligence Summarization (Direct Audio)
Transcripts are bypassed in the primary flow. The pipeline uses **Google Gemini 2.5 Flash** to perform native audio analysis, listening for both semantic content and acoustic signals (tone, pacing, conviction).

The engine generates:
- **Episode Intelligence Profile**
- **Signal Snapshot** (compact scoring grid)
- **Reality Layer** (hard claims only)
- **Forward Projections** (risk layer)
- **Mechanism Analysis**
- **Emotional Conviction** (acoustic tone analysis)
- **Disconfirming Signals**
- **Final Intelligence Take** (strategic classification)

**Output:**
- Saved as Markdown in `/briefs/`
- Optionally emailed
- Fully structured JSON retained internally for scoring and longitudinal analysis

## Output Characteristics (BKM v2)
Each episode brief:
- 400–700 words
- <3 minutes reading time
- Compact scoring tables
- Max 6 bullets per section
- Strict compression of thesis (1 sentence)
- Final strategic classification required

*TheBrief is optimized for cognitive efficiency, not verbosity.*

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kvwilliamson/TheBrief.git
   cd TheBrief
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and configure:
   - `YOUTUBE_API_KEY` – Required for discovery
   - `GOOGLE_AI_API_KEY` – Required for Gemini native audio analysis
   - `OPENAI_API_KEY` – Optional Whisper fallback

   *(Optional) Email configuration:*
   - `SEND_EMAIL=true`
   - `EMAIL_TO`
   - `EMAIL_FROM`
   - `SMTP_HOST`
   - `SMTP_PASSWORD` (App Password recommended)

4. **Curate channels:**
   Edit `channels.json` with channel names and official YouTube Channel IDs.

## Running the System

### Recommended: Dashboard UI
```bash
streamlit run app.py
```
Opens a local web interface to:
- View briefs
- Manage tracked channels
- Trigger pipeline manually

### Headless Mode
```bash
python main.py
```
Runs the full discovery → extraction → transcription → intelligence pipeline.

## GitHub Actions Automation

A scheduled workflow runs daily at **6:00 AM UTC**.

**Configuration file:** `.github/workflows/daily_brief.yml`

**To enable:**
1. Go to Repository → **Settings** → **Secrets and variables** → **Actions**
2. Add:
   - `YOUTUBE_API_KEY`
   - `GOOGLE_AI_API_KEY`
   - `OPENAI_API_KEY`
   - *(Optional)* `SMTP_PASSWORD`

The cron job processes new long-form episodes, generates intelligence briefs via native audio analysis, and commits Markdown files into `/briefs/`—fully automated.

## Design Intent

TheBrief is built for:
- High-agency thinkers
- Portfolio decision-makers
- Researchers
- Signal-oriented operators

*It is not designed for entertainment recaps.*

It is designed to answer:
- What was claimed?
- How strong is the evidence?
- Where is the narrative pressure?
- What would break the thesis?
- Does this matter?

## Roadmap (Future Expansion)

Planned architecture supports:
- Prediction tracking & calibration scoring
- Channel Credibility Index
- Base-rate modeling for financial claims
- Longitudinal narrative drift detection
- Automated fact-check layering
