"""
Shared data models for the API and model inference.
"""

from typing import List, Optional
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: Optional[str] = "You are a helpful assistant."
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    total_tokens: int
    inference_time: float 