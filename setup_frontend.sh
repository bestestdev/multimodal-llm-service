#!/bin/bash

# Multimodal LLM Service Frontend setup script
set -e

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed. Please install Node.js and npm."
    exit 1
fi

echo "Setting up the React frontend..."

# Change to the frontend directory
cd frontend

# Install dependencies
echo "Installing frontend dependencies..."
npm install

# Build the frontend
echo "Building the frontend..."
npm run build

echo "Frontend setup complete!"
echo "You can now run the service with ./run.sh" 