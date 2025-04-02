"""
WebSocket API routes for realtime chat functionality.
"""

import json
import asyncio
from typing import Dict, List, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, ValidationError

from models.inference import ModelManager

router = APIRouter(tags=["websocket"])

# Initialize model manager
model_manager = ModelManager()

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

# Initialize connection manager
manager = ConnectionManager()

@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for realtime chat.
    
    Supports both text and audio data.
    """
    # Generate a unique client ID (in production, use proper authentication)
    client_id = f"client_{id(websocket)}"
    
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type", "text")
                
                if message_type == "text":
                    # Handle text message
                    response = await model_manager.generate_text_response(
                        messages=message_data.get("messages", []),
                        system_prompt=message_data.get("system_prompt", "You are a helpful assistant."),
                        max_tokens=message_data.get("max_tokens", 1024),
                        temperature=message_data.get("temperature", 0.7),
                        stream=True
                    )
                    
                    # Stream the response
                    async for token in response:
                        await manager.send_message(
                            json.dumps({"type": "token", "content": token}),
                            client_id
                        )
                        
                    # Send end of stream marker
                    await manager.send_message(
                        json.dumps({"type": "end"}),
                        client_id
                    )
                    
                elif message_type == "audio":
                    # Handle base64 encoded audio data
                    audio_data = message_data.get("audio_data")
                    if not audio_data:
                        await manager.send_message(
                            json.dumps({"type": "error", "content": "No audio data provided"}),
                            client_id
                        )
                        continue
                    
                    # Process audio data
                    response = await model_manager.generate_audio_response_from_base64(
                        audio_base64=audio_data,
                        system_prompt=message_data.get("system_prompt", "You are a helpful assistant."),
                        max_tokens=message_data.get("max_tokens", 1024),
                        temperature=message_data.get("temperature", 0.7),
                        stream=True
                    )
                    
                    # Stream the response
                    async for token in response:
                        await manager.send_message(
                            json.dumps({"type": "token", "content": token}),
                            client_id
                        )
                        
                    # Send end of stream marker
                    await manager.send_message(
                        json.dumps({"type": "end"}),
                        client_id
                    )
                    
                else:
                    await manager.send_message(
                        json.dumps({"type": "error", "content": f"Unsupported message type: {message_type}"}),
                        client_id
                    )
                    
            except json.JSONDecodeError:
                await manager.send_message(
                    json.dumps({"type": "error", "content": "Invalid JSON format"}),
                    client_id
                )
            except ValidationError as e:
                await manager.send_message(
                    json.dumps({"type": "error", "content": f"Validation error: {str(e)}"}),
                    client_id
                )
            except Exception as e:
                await manager.send_message(
                    json.dumps({"type": "error", "content": f"Error processing request: {str(e)}"}),
                    client_id
                )
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        manager.disconnect(client_id)
        raise HTTPException(status_code=500, detail=f"WebSocket error: {str(e)}") 