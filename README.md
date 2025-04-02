# Multimodal LLM Service

A realtime voice/text communication service powered by Ultravox, enabling natural conversations with an AI through both voice and text input.

## Overview

This project implements a local multimodal LLM service that can:
- Process and understand spoken audio input
- Handle text-based prompts
- Support realtime conversations
- Provide a web interface and API access

The service uses [Ultravox](https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_1-8b), a multimodal Speech LLM built around a pretrained Llama3.1-8B-Instruct and whisper-large-v3-turbo backbone.

## Requirements

- Python 3.10+
- NVIDIA GPU with at least 12GB VRAM
- NVIDIA drivers and CUDA toolkit installed
- Modern web browser for the web interface

## Tech Stack

### Backend
- **FastAPI**: High-performance API framework with WebSocket support
- **Uvicorn**: ASGI server for running the FastAPI application
- **HuggingFace Transformers**: For loading and running the Ultravox model
- **PyTorch**: Deep learning framework
- **Librosa**: Audio processing library

### Frontend
- **React**: For building the web interface
- **Web Audio API**: For audio capture and streaming
- **WebSockets**: For realtime communication

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/multimodal-llm-service.git
cd multimodal-llm-service
```

2. Run the setup script to install dependencies and set up the frontend:
```bash
./setup_frontend.sh
```

This script will:
- Create and activate a Python virtual environment
- Install Python dependencies from requirements.txt
- Install frontend dependencies

3. Start the service using the provided run script:
```bash
./run.sh
```

The service will be available at http://localhost:8000. The web interface can be accessed at http://localhost:8000/app.

Alternatively, you can set up manually:

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the service:
```bash
python src/main.py
```

## API Usage

### REST API

#### Text Input
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello, how are you?"}]}'
```

#### Audio Input
```bash
curl -X POST "http://localhost:8000/api/chat/audio" \
  -F "audio=@path/to/audio.wav" \
  -F "system_prompt=You are a helpful assistant."
```

### WebSocket API

For realtime conversation, connect to:
```
ws://localhost:8000/ws/chat
```

See the API documentation for detailed usage examples.

## Project Structure

```
multimodal-llm-service/
├── src/
│   ├── main.py               # Application entry point
│   ├── api/                  # API endpoints
│   ├── models/               # Model loading and inference
│   ├── audio/                # Audio processing utilities
│   └── utils/                # Helper functions
├── frontend/                 # React frontend
├── tests/                    # Test suite
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Performance Considerations

- The model requires approximately 8GB of VRAM to load in BF16 precision
- Initial response latency (TTFT) is approximately 150ms
- Processing speed is around 50-100 tokens per second on an NVIDIA GPU

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
