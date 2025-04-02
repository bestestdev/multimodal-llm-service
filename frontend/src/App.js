import React, { useState, useEffect, useRef } from 'react';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [socket, setSocket] = useState(null);
  const audioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  
  // Connect to WebSocket
  useEffect(() => {
    const socketConnection = new WebSocket(`ws://${window.location.host}/ws/chat`);
    
    socketConnection.onopen = () => {
      console.log('WebSocket connection established');
      setIsConnected(true);
    };
    
    socketConnection.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
          // Handle streaming token
          setMessages(msgs => {
            const newMsgs = [...msgs];
            const lastMsg = newMsgs[newMsgs.length - 1];
            if (lastMsg && lastMsg.role === 'assistant') {
              // Append to existing assistant message
              lastMsg.content += data.content;
            } else {
              // Create a new assistant message
              newMsgs.push({ role: 'assistant', content: data.content });
            }
            return newMsgs;
          });
        } else if (data.type === 'error') {
          console.error('Error from server:', data.content);
        }
      } catch (e) {
        console.error('Error parsing WebSocket message:', e);
      }
    };
    
    socketConnection.onclose = () => {
      console.log('WebSocket connection closed');
      setIsConnected(false);
    };
    
    socketConnection.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    setSocket(socketConnection);
    
    return () => {
      socketConnection.close();
    };
  }, []);
  
  // Send text message
  const sendMessage = () => {
    if (!input.trim() || !socket) return;
    
    const userMessage = { role: 'user', content: input };
    setMessages(msgs => [...msgs, userMessage]);
    
    const messageData = {
      type: 'text',
      messages: [...messages, userMessage],
      system_prompt: 'You are a helpful assistant.'
    };
    
    socket.send(JSON.stringify(messageData));
    setInput('');
  };
  
  // Start recording audio
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        sendAudioMessage(audioBlob);
      };
      
      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Error accessing microphone:', err);
    }
  };
  
  // Stop recording and send audio
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (audioRef.current) {
        audioRef.current.getTracks().forEach(track => track.stop());
      }
    }
  };
  
  // Send audio message
  const sendAudioMessage = async (audioBlob) => {
    if (!socket) return;
    
    // Convert blob to base64
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = () => {
      const base64Audio = reader.result.split(',')[1]; // Remove data URL prefix
      
      setMessages(msgs => [...msgs, { role: 'user', content: '🎤 [Audio Message]' }]);
      
      const messageData = {
        type: 'audio',
        audio_data: base64Audio,
        system_prompt: 'You are a helpful assistant.'
      };
      
      socket.send(JSON.stringify(messageData));
    };
  };
  
  return (
    <div className="app">
      <header>
        <h1>Multimodal LLM Chat</h1>
        <div className="connection-status">
          {isConnected ? 'Connected ✅' : 'Disconnected ❌'}
        </div>
      </header>
      
      <div className="chat-container">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
      </div>
      
      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type a message..."
          disabled={!isConnected}
        />
        <button onClick={sendMessage} disabled={!isConnected}>Send</button>
        <button 
          onClick={isRecording ? stopRecording : startRecording}
          className={isRecording ? 'recording' : ''}
          disabled={!isConnected}
        >
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </button>
      </div>
    </div>
  );
}

export default App; 