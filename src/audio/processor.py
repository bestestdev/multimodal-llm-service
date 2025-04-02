"""
Audio processing utilities for the multimodal LLM service.
"""

import os
import base64
import tempfile
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from typing import Tuple, Optional

class AudioProcessor:
    """Audio processing utilities for handling WAV files and audio data."""
    
    def __init__(self, target_sr: int = 16000):
        """
        Initialize the audio processor.
        
        Args:
            target_sr: Target sampling rate for the audio
        """
        self.target_sr = target_sr
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and resample if necessary.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sampling_rate)
        """
        audio, sr = librosa.load(file_path, sr=self.target_sr)
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, file_path: str, sr: int = None) -> str:
        """
        Save audio data to a file.
        
        Args:
            audio: Audio data
            file_path: Path to save the audio file
            sr: Sampling rate
            
        Returns:
            Path to the saved file
        """
        if sr is None:
            sr = self.target_sr
            
        sf.write(file_path, audio, sr)
        return file_path
    
    def encode_to_base64(self, audio: np.ndarray, sr: int = None) -> str:
        """
        Encode audio data to base64.
        
        Args:
            audio: Audio data
            sr: Sampling rate
            
        Returns:
            Base64-encoded audio
        """
        if sr is None:
            sr = self.target_sr
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio, sr)
            
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()
                
            os.unlink(temp_path)
            return base64.b64encode(audio_bytes).decode("utf-8")
    
    def decode_from_base64(self, base64_str: str) -> Tuple[np.ndarray, int]:
        """
        Decode audio from base64 to numpy array.
        
        Args:
            base64_str: Base64-encoded audio string
            
        Returns:
            Tuple of (audio_data, sampling_rate)
        """
        audio_bytes = base64.b64decode(base64_str)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
            
        try:
            audio, sr = self.load_audio(temp_path)
            return audio, sr
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def get_audio_duration(self, audio: np.ndarray, sr: int = None) -> float:
        """
        Get the duration of an audio signal in seconds.
        
        Args:
            audio: Audio data
            sr: Sampling rate
            
        Returns:
            Duration in seconds
        """
        if sr is None:
            sr = self.target_sr
            
        return librosa.get_duration(y=audio, sr=sr)
    
    def trim_silence(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Trim silence from the beginning and end of an audio signal.
        
        Args:
            audio: Audio data
            sr: Sampling rate
            
        Returns:
            Trimmed audio data
        """
        if sr is None:
            sr = self.target_sr
            
        trimmed_audio, _ = librosa.effects.trim(audio)
        return trimmed_audio 