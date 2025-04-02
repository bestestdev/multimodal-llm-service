#!/bin/bash

# Multimodal LLM Service startup script
set -e

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    touch .deps_installed
fi

# Change to the src directory
cd src

# Start the server
echo "Starting the Multimodal LLM Service..."
python main.py 