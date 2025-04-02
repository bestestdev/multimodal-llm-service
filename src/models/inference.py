"""
Model management and inference for the Ultravox model.
"""

import time
import base64
import asyncio
import tempfile
import os
import gc
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import numpy as np
import torch
import librosa
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from pydantic import BaseModel

from models.schemas import Message, ChatResponse

class ModelManager:
    """Manager for loading and running the Ultravox model."""
    
    def __init__(self):
        """Initialize the model manager and load the model."""
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.sampling_rate = 16000  # Required sampling rate for Ultravox
        self.model_id = 'fixie-ai/ultravox-v0_5-llama-3_1-8b'
        
        # Control how many layers to offload to CPU
        # Can be controlled via environment variable ULTRAVOX_CPU_OFFLOAD_LAYERS
        self.cpu_offload_layers = int(os.environ.get("ULTRAVOX_CPU_OFFLOAD_LAYERS", "0"))
        print(f"Will offload {self.cpu_offload_layers} layers to CPU")
        
        # Whether to use 4-bit quantization (bitsandbytes)
        self.use_4bit = os.environ.get("ULTRAVOX_USE_4BIT", "").lower() in ("1", "true", "yes")
        if self.use_4bit:
            print("Using 4-bit quantization to reduce memory usage")
        
        # Set up CUDA memory management
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            # Print available GPU memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            print(f"Available GPU memory: {free_memory / (1024**3):.2f} GB")
        
    async def load_model(self):
        """Load the Ultravox model asynchronously."""
        if self.model_loaded:
            return
        
        # Clear cache before loading
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # This is run in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)
        
    def _load_model_sync(self):
        """Synchronous model loading function."""
        if self.model_loaded:
            return
            
        try:
            print("Loading Ultravox model...")
            
            # Create offload directory if it doesn't exist
            offload_folder = "offload"
            os.makedirs(offload_folder, exist_ok=True)
            
            # Prepare loading options
            load_options = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True  # Reduces CPU memory usage during loading
            }
            
            # Add 4-bit quantization if enabled
            if self.use_4bit and self.device == "cuda":
                print("Setting up 4-bit quantization with bitsandbytes")
                load_options["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                load_options["torch_dtype"] = torch.float16
            
            # For Ultravox, we need to use the pipeline without device_map since it's causing issues
            # with the custom model architecture
            print("Loading model with standard pipeline")
            
            # Remove any device map settings as they're causing issues
            if "ACCELERATE_OFFLOAD_PARAM_FRACTION" in os.environ:
                del os.environ["ACCELERATE_OFFLOAD_PARAM_FRACTION"]
            
            # Use lower precision to save memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Setting default cuda dtype to float16")
                torch.set_default_dtype(torch.float16)
                
                # Set up additional CUDA memory optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                
                # Use with_fp16() instead of directly setting tensor type which can cause issues
                print("Enabling FP16 for inference")
            
            # If needed, add specific arguments for Llama-family models since Ultravox is based on Llama
            if self.model_id.lower().find('llama') != -1:
                load_options['use_flash_attention_2'] = True
                print("Enabling Flash Attention 2 for faster inference")
            
            # Load the model with simple pipeline configuration
            self.pipe = transformers.pipeline(
                model=self.model_id,
                **load_options
            )
            
            # We still need the tokenizer for some operations
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Extract model reference from pipeline
            self.model = self.pipe.model
            
            # Enable gradient checkpointing if available (saves memory during inference)
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing")
            
            self.model_loaded = True
            print(f"Model loaded successfully")
            
            # Print memory usage after loading
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"GPU memory allocated: {allocated:.2f} GB (max: {max_allocated:.2f} GB)")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    async def generate_text_response(
        self, 
        messages: List[Message],
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[ChatResponse, AsyncGenerator[str, None]]:
        """
        Generate a response from text input.
        
        Args:
            messages: List of conversation messages
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Either a ChatResponse object or a streaming generator
        """
        # Ensure model is loaded
        await self.load_model()
        
        # Format the conversation history for the model
        turns = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add user messages
        for msg in messages:
            turns.append({
                "role": msg.role,
                "content": msg.content
            })
        
        start_time = time.time()
        
        # Format conversation for the model
        conversation = {"turns": turns}
        
        if stream:
            # Return a streaming generator
            return self._stream_text_response(conversation, max_tokens, temperature)
        else:
            # Generate the full response at once
            try:
                # Process with the pipeline
                response = await asyncio.to_thread(
                    self.pipe,
                    conversation,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0)
                )
                
                inference_time = time.time() - start_time
                
                # The pipeline handles tokenization and generation
                return ChatResponse(
                    response=response[0]["generated_text"] if isinstance(response, list) else response["generated_text"],
                    total_tokens=0,  # We don't have access to token count directly from pipeline
                    inference_time=inference_time
                )
            finally:
                # Clean up memory after inference
                self._cleanup_memory()
    
    async def _stream_text_response(
        self,
        conversation: Dict,
        max_tokens: int,
        temperature: float
    ) -> AsyncGenerator[str, None]:
        """Stream the text response token by token."""
        try:
            # Note: The pipeline doesn't support streaming out of the box
            # For now, we'll just get the full response and yield it
            
            response = await asyncio.to_thread(
                self.pipe,
                conversation,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0)
            )
            
            # Extract and yield the response text
            response_text = response[0]["generated_text"] if isinstance(response, list) else response["generated_text"]
            yield response_text
        finally:
            # Clean up memory after inference
            self._cleanup_memory()
    
    async def generate_audio_response(
        self,
        audio_path: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[ChatResponse, AsyncGenerator[str, None]]:
        """
        Generate a response from audio input.
        
        Args:
            audio_path: Path to the audio file
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Either a ChatResponse object or a streaming generator
        """
        # Ensure model is loaded
        await self.load_model()
        
        # Load and preprocess the audio
        audio, sr = await asyncio.to_thread(librosa.load, audio_path, sr=self.sampling_rate)
        
        # Format the conversation turns for audio input
        turns = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        start_time = time.time()
        
        # Create the multimodal conversation input
        conversation = {
            "audio": audio,
            "turns": turns,
            "sampling_rate": self.sampling_rate
        }
        
        if stream:
            # Return a streaming generator
            return self._stream_audio_response(conversation, max_tokens, temperature)
        else:
            # Generate the full response at once
            try:
                # Process the input with the pipeline as shown in Hugging Face example
                response = await asyncio.to_thread(
                    self.pipe,
                    conversation,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0)
                )
                
                inference_time = time.time() - start_time
                
                # Extract response text from pipeline output
                return ChatResponse(
                    response=response[0]["generated_text"] if isinstance(response, list) else response["generated_text"],
                    total_tokens=0,  # We don't have access to token count directly
                    inference_time=inference_time
                )
            finally:
                # Clean up memory after inference
                self._cleanup_memory()
    
    async def _stream_audio_response(
        self,
        conversation: Dict,
        max_tokens: int,
        temperature: float
    ) -> AsyncGenerator[str, None]:
        """Stream the audio response token by token."""
        try:
            # Similar to text streaming, the pipeline doesn't have streaming capabilities
            # Just get the full response and yield it
            
            response = await asyncio.to_thread(
                self.pipe,
                conversation,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0)
            )
            
            # Extract and yield the response text
            response_text = response[0]["generated_text"] if isinstance(response, list) else response["generated_text"]
            yield response_text
        finally:
            # Clean up memory after inference
            self._cleanup_memory()
    
    async def generate_audio_response_from_base64(
        self,
        audio_base64: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[ChatResponse, AsyncGenerator[str, None]]:
        """
        Generate a response from base64-encoded audio.
        
        Args:
            audio_base64: Base64-encoded audio data
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Either a ChatResponse object or a streaming generator
        """
        # Decode the base64 audio to a temporary file
        audio_bytes = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            temp_file.write(audio_bytes)
        
        try:
            # Process the audio file
            return await self.generate_audio_response(
                audio_path=temp_filename,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename) 