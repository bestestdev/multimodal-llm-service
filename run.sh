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

# Set model configuration for better memory management
# Instead of CPU offloading, we'll use 4-bit quantization which is more compatible
export ULTRAVOX_CPU_OFFLOAD_LAYERS=0  # Disable CPU offloading as it's causing issues
export ULTRAVOX_USE_4BIT=true         # Use 4-bit quantization for memory savings
echo "Using 4-bit quantization for model loading"

# PyTorch CUDA memory management settings - help prevent OOM errors
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
echo "Configured PyTorch CUDA memory allocator"

# Start the server
echo "Starting the Multimodal LLM Service..."
python main.py 