#!/bin/bash
# TheBrief Startup Script

# Navigate to the project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    /opt/homebrew/bin/python3.13 -m venv venv
    ./venv/bin/pip install -r requirements.txt
fi

# Auto-update yt-dlp to stay ahead of YouTube anti-bot changes
echo "🔄 Checking for yt-dlp updates..."
./venv/bin/pip install -U yt-dlp --quiet 2>/dev/null
echo "✅ yt-dlp $(./venv/bin/python3 -m yt_dlp --version)"

echo "🚀 Starting TheBrief Dashboard..."
./venv/bin/streamlit run app.py
