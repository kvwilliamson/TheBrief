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

echo "🚀 Starting TheBrief Dashboard..."
./venv/bin/streamlit run app.py
