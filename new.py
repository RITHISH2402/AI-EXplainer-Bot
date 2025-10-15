from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from flask import Flask, request, render_template_string, jsonify
import os
import re
import random
import time
from datetime import datetime

# --------- Configuration ---------
MODEL_NAME = "deepset/roberta-base-squad2"
KNOWLEDGE_BASE_PATH = "knowledge_base.txt"

# --------- Model Loading ---------
print("Loading AI model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    print("✅ AI model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# --------- Flask App ---------
app = Flask(__name__)
app.secret_key = 'ai_tutor_secret_key'

# Interactive conversation memory
conversation_history = []
user_interests = set()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🤖 AI Learning Companion</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
            transition: all 0.3s ease;
        }
        body.dark-theme {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        }
        .container {
            max-width: 900px; margin: 0 auto; background: white;
            border-radius: 20px; box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            overflow: hidden; height: 90vh; display: flex; flex-direction: column;
            transition: all 0.3s ease;
        }
        body.dark-theme .container {
            background: #2c3e50;
            color: #ecf0f1;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white; padding: 25px; text-align: center;
            flex-shrink: 0; position: relative;
        }
        body.dark-theme .header {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        }
        .header h1 { 
            font-size: 28px; margin: 0; 
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .header-subtitle {
            font-size: 14px; opacity: 0.9; margin-top: 5px;
        }
        .ai-avatar {
            position: absolute; left: 25px; top: 50%; transform: translateY(-50%);
            width: 50px; height: 50px; background: rgba(255,255,255,0.2);
            border-radius: 50%; display: flex; align-items: center;
            justify-content: center; font-size: 24px;
        }
        .theme-toggle {
            position: absolute; right: 25px; top: 50%; transform: translateY(-50%);
            background: rgba(255,255,255,0.2); border: none; color: white;
            padding: 10px 15px; border-radius: 25px; cursor: pointer;
            font-size: 14px; transition: all 0.3s ease;
            display: flex; align-items: center; gap: 8px;
        }
        .theme-toggle:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-50%) scale(1.05);
        }
        
        .chat-container { 
            flex: 1; display: flex; flex-direction: column; 
            min-height: 0; background: #f8f9fa;
            transition: all 0.3s ease;
        }
        body.dark-theme .chat-container {
            background: #34495e;
        }
        
        .chat-messages { 
            flex: 1; padding: 25px; 
            overflow-y: auto; background: #f8f9fa; 
            display: flex; flex-direction: column;
            min-height: 0;
            transition: all 0.3s ease;
        }
        body.dark-theme .chat-messages {
            background: #34495e;
            color: #ecf0f1;
        }
        
        .message { 
            margin-bottom: 20px; display: flex; align-items: flex-start;
            gap: 15px; animation: fadeIn 0.4s ease-out;
            flex-shrink: 0;
        }
        .message.user { flex-direction: row-reverse; }
        .avatar { 
            width: 45px; height: 45px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; color: white; flex-shrink: 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .avatar:hover { transform: scale(1.05); }
        .avatar.bot { 
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        }
        .avatar.user { 
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        }
        .message-content { 
            max-width: 75%; padding: 18px; border-radius: 20px;
            line-height: 1.6; word-wrap: break-word;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .bot .message-content { 
            background: white; border: 2px solid #e1e8ed;
            border-bottom-left-radius: 5px;
        }
        body.dark-theme .bot .message-content {
            background: #2c3e50;
            border-color: #34495e;
            color: #ecf0f1;
        }
        .user .message-content { 
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white; border-bottom-right-radius: 5px;
        }
        
        .input-area { 
            padding: 25px; border-top: 2px solid #e1e8ed;
            background: white; flex-shrink: 0;
            position: relative;
            transition: all 0.3s ease;
        }
        body.dark-theme .input-area {
            background: #2c3e50;
            border-color: #34495e;
        }
        .input-group { display: flex; gap: 15px; align-items: center; }
        input[type="text"] { 
            flex: 1; padding: 15px 20px; border: 2px solid #e1e8ed;
            border-radius: 30px; font-size: 16px; outline: none;
            transition: all 0.3s ease; background: #f8f9fa;
        }
        body.dark-theme input[type="text"] {
            background: #34495e;
            border-color: #4a6572;
            color: #ecf0f1;
        }
        input[type="text"]:focus { 
            border-color: #3498db; background: white;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        body.dark-theme input[type="text"]:focus {
            background: #2c3e50;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }
        button { 
            padding: 15px 30px; background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white; border: none; border-radius: 30px; cursor: pointer;
            font-size: 16px; transition: all 0.3s ease; font-weight: 600;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }
        button:hover { 
            transform: translateY(-2px); box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
        }
        
        /* Interactive Elements */
        .quick-actions { 
            display: flex; gap: 10px; margin-top: 15px;
            flex-wrap: wrap; justify-content: center;
        }
        .quick-action { 
            background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
            border: none; padding: 10px 20px; border-radius: 25px; 
            font-size: 14px; cursor: pointer; transition: all 0.3s ease;
            font-weight: 500; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        body.dark-theme .quick-action {
            background: linear-gradient(135deg, #34495e 0%, #4a6572 100%);
            color: #ecf0f1;
        }
        .quick-action:hover { 
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white; transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }
        
        .interactive-buttons {
            display: flex; gap: 10px; margin: 15px 0; flex-wrap: wrap;
        }
        .interactive-btn {
            background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
            color: white; border: none; padding: 8px 16px;
            border-radius: 20px; font-size: 12px; cursor: pointer;
            transition: all 0.3s ease; font-weight: 500;
        }
        .interactive-btn:hover {
            transform: scale(1.05); box-shadow: 0 4px 12px rgba(155, 89, 182, 0.3);
        }
        
        /* Enhanced Answer Styling */
        .answer-box { 
            background: white; border-radius: 15px; padding: 25px;
            margin: 10px 0; border-left: 5px solid #3498db;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            animation: slideIn 0.5s ease-out;
            transition: all 0.3s ease;
        }
        body.dark-theme .answer-box {
            background: #2c3e50;
            border-left-color: #3498db;
        }
        .topic-badge { 
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white; padding: 8px 20px; border-radius: 20px; 
            font-size: 14px; font-weight: bold; display: inline-block; 
            margin-bottom: 20px; box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }
        
        .info-card {
            background: linear-gradient(135deg, #e8f4fd 0%, #d4e6f1 100%);
            padding: 20px; border-radius: 12px; margin: 15px 0;
            border-left: 4px solid #2980b9; position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        body.dark-theme .info-card {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            border-left-color: #3498db;
        }
        .info-card::before {
            content: ''; position: absolute; top: 0; left: 0;
            width: 100%; height: 3px; 
            background: linear-gradient(90deg, #3498db, #2980b9);
        }
        
        .key-points-card {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            padding: 20px; border-radius: 12px; margin: 15px 0;
            border-left: 4px solid #f39c12;
            transition: all 0.3s ease;
        }
        body.dark-theme .key-points-card {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            border-left-color: #f39c12;
        }
        
        .example-card {
            background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%);
            padding: 20px; border-radius: 12px; margin: 15px 0;
            border: 2px dashed #3498db;
            transition: all 0.3s ease;
        }
        body.dark-theme .example-card {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            border-color: #3498db;
        }
        
        .fun-fact-card {
            background: linear-gradient(135deg, #d4edda 0%, #c8e6c9 100%);
            padding: 18px; border-radius: 12px; margin: 15px 0;
            border-left: 4px solid #27ae60; position: relative;
            transition: all 0.3s ease;
        }
        body.dark-theme .fun-fact-card {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            border-left-color: #27ae60;
        }
        .fun-fact-card::after {
            content: '💡'; position: absolute; top: 15px; right: 15px;
            font-size: 20px; opacity: 0.3;
        }
        
        .challenge-box {
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            padding: 20px; border-radius: 12px; margin: 15px 0;
            border: 2px solid #e17055; text-align: center;
            transition: all 0.3s ease;
        }
        body.dark-theme .challenge-box {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            border-color: #e17055;
        }
        .challenge-btn {
            background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
            color: white; border: none; padding: 10px 25px;
            border-radius: 25px; margin-top: 10px; cursor: pointer;
            font-weight: 600; transition: all 0.3s ease;
        }
        .challenge-btn:hover {
            transform: scale(1.05); box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        }
        
        .progress-tracker {
            background: rgba(52, 152, 219, 0.1); padding: 15px;
            border-radius: 10px; margin: 15px 0; text-align: center;
            border: 2px solid rgba(52, 152, 219, 0.2);
            transition: all 0.3s ease;
        }
        body.dark-theme .progress-tracker {
            background: rgba(52, 152, 219, 0.2);
            border-color: rgba(52, 152, 219, 0.3);
        }
        
        /* Animations */
        .typing-indicator { 
            display: flex; gap: 6px; padding: 15px; 
            align-items: center;
        }
        .typing-dot { 
            width: 10px; height: 10px; border-radius: 50%;
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            animation: typing 1.4s infinite;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        
        .welcome-message { 
            text-align: center; padding: 30px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border-radius: 15px; margin: 10px 0;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        body.dark-theme .welcome-message {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        }
        
        .unrelated-warning { 
            background: linear-gradient(135deg, #f8d7da 0%, #f5b7b1 100%);
            color: #721c24; padding: 20px; border-radius: 12px;
            margin: 10px 0; border-left: 5px solid #dc3545;
            text-align: center;
            transition: all 0.3s ease;
        }
        body.dark-theme .unrelated-warning {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            border-left-color: #dc3545;
        }
        
        /* Custom scrollbar */
        .chat-messages::-webkit-scrollbar { width: 10px; }
        .chat-messages::-webkit-scrollbar-track { 
            background: #f1f1f1; border-radius: 10px;
        }
        body.dark-theme .chat-messages::-webkit-scrollbar-track {
            background: #2c3e50;
        }
        .chat-messages::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            border-radius: 10px;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-30px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .pulse { animation: pulse 2s infinite; }
        .bounce { animation: bounce 1s infinite; }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .learning-path {
            background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
            color: white; padding: 15px; border-radius: 12px;
            margin: 15px 0; text-align: center;
        }
        body.dark-theme .learning-path {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="ai-avatar">🤖</div>
            <button class="theme-toggle" onclick="toggleTheme()">
                <span id="themeIcon">🌙</span>
                <span id="themeText">Dark Mode</span>
            </button>
            <h1>AI Learning Companion</h1>
            <div class="header-subtitle">Your Personal Guide to Artificial Intelligence</div>
        </div>
        
        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    <h3>🎉 Welcome to Your AI Learning Journey!</h3>
                    <p>I'm your personal AI tutor, here to make learning about artificial intelligence fun, interactive, and engaging!</p>
                    <div style="margin-top: 15px; font-size: 14px; opacity: 0.9;">
                        ✨ <strong>Ready to explore the future of technology?</strong>
                    </div>
                </div>
                <div class="message bot">
                    <div class="avatar bot">AI</div>
                    <div class="message-content">
                        <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; padding: 20px; border-radius: 15px;">
                            <strong>🚀 Let's Start Your AI Adventure!</strong><br><br>
                            I can help you:<br>
                            • 🤔 Understand complex AI concepts simply<br>
                            • 💡 Discover real-world applications<br>
                            • 🎯 Solve interactive challenges<br>
                            • 📚 Build your learning path<br><br>
                            
                            <div class="interactive-buttons">
                                <button class="interactive-btn" onclick="askQuestion('What is machine learning?')">Start with ML</button>
                                <button class="interactive-btn" onclick="askQuestion('Show me AI applications')">See Applications</button>
                                <button class="interactive-btn" onclick="askQuestion('Take a quiz')">Quick Quiz</button>
                                <button class="interactive-btn" onclick="askQuestion('Learning path')">My Learning Path</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <div class="input-group">
                    <input type="text" id="messageInput" placeholder="Ask me anything about AI or type 'help' for options..." autofocus>
                    <button onclick="sendMessage()">Send 🚀</button>
                </div>
                <div class="quick-actions">
                    <div class="quick-action" onclick="askQuestion('What is machine learning?')">🤖 ML Basics</div>
                    <div class="quick-action" onclick="askQuestion('What is CNN?')">👁️ Computer Vision</div>
                    <div class="quick-action" onclick="askQuestion('What is AI ethics?')">⚖️ AI Ethics</div>
                    <div class="quick-action" onclick="askQuestion('What is deep learning?')">🧠 Deep Learning</div>
                    <div class="quick-action" onclick="askQuestion('What is NLP?')">💬 NLP</div>
                    <div class="quick-action" onclick="askQuestion('quiz')">🎯 Take a Quiz</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let messageCount = 0;
        let userLevel = 'beginner';
        let isDarkTheme = false;
        
        // Theme management
        function toggleTheme() {
            isDarkTheme = !isDarkTheme;
            const body = document.body;
            const themeIcon = document.getElementById('themeIcon');
            const themeText = document.getElementById('themeText');
            
            if (isDarkTheme) {
                body.classList.add('dark-theme');
                themeIcon.textContent = '☀️';
                themeText.textContent = 'Light Mode';
                localStorage.setItem('theme', 'dark');
            } else {
                body.classList.remove('dark-theme');
                themeIcon.textContent = '🌙';
                themeText.textContent = 'Dark Mode';
                localStorage.setItem('theme', 'light');
            }
        }
        
        // Load saved theme
        function loadTheme() {
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark') {
                isDarkTheme = true;
                document.body.classList.add('dark-theme');
                document.getElementById('themeIcon').textContent = '☀️';
                document.getElementById('themeText').textContent = 'Light Mode';
            }
        }
        
        function addMessage(content, isUser = false) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const avatar = document.createElement('div');
            avatar.className = `avatar ${isUser ? 'user' : 'bot'}`;
            avatar.textContent = isUser ? 'You' : 'AI';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = content;
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            if (isUser) {
                messageCount++;
                // Update user level based on engagement
                if (messageCount > 10) userLevel = 'intermediate';
                if (messageCount > 25) userLevel = 'advanced';
            }
        }
        
        function showTyping() {
            const chatMessages = document.getElementById('chatMessages');
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot';
            typingDiv.id = 'typingIndicator';
            
            typingDiv.innerHTML = `
                <div class="avatar bot">AI</div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <span style="margin-left: 10px; color: #666; font-size: 14px;">Thinking...</span>
                    </div>
                </div>
            `;
            
            chatMessages.appendChild(typingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function hideTyping() {
            const typingIndicator = document.getElementById('typingIndicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, true);
            input.value = '';
            
            showTyping();
            
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    question: message,
                    message_count: messageCount,
                    user_level: userLevel
                })
            })
            .then(response => response.json())
            .then(data => {
                hideTyping();
                if (data.success) {
                    // Add slight delay for more natural conversation flow
                    setTimeout(() => {
                        addMessage(data.answer, false);
                    }, 500);
                } else {
                    addMessage('Sorry, I encountered an error. Please try again.', false);
                }
            })
            .catch(error => {
                hideTyping();
                addMessage('Sorry, I encountered an error. Please try again.', false);
                console.error('Error:', error);
            });
        }
        
        function askQuestion(question) {
            document.getElementById('messageInput').value = question;
            sendMessage();
        }

        function checkAnswer(selected, correct, explanation) {
            const resultDiv = document.getElementById('quizResult');
            if (selected === correct) {
                resultDiv.innerHTML = `<div style="color: green; font-weight: bold;">✅ Correct! Well done!</div><div>${explanation}</div>`;
            } else {
                resultDiv.innerHTML = `<div style="color: red; font-weight: bold;">❌ Not quite right. The correct answer is option ${correct + 1}.</div><div>${explanation}</div>`;
        }
    }

        function submitChallenge(selected, correct, explanation) {
            const resultDiv = document.getElementById('challengeResult');
            if (selected === correct) {
                resultDiv.innerHTML = `<div style="color: green; font-weight: bold;">✅ Correct! Well done!</div><div>${explanation}</div>`;
            } else {
                resultDiv.innerHTML = `<div style="color: red; font-weight: bold;">❌ Not quite right. The correct answer is option ${correct + 1}.</div><div>${explanation}</div>`;
    }
}
        
        function triggerAnimation(elementId) {
            const element = document.getElementById(elementId);
            if (element) {
                element.classList.add('pulse');
                setTimeout(() => element.classList.remove('pulse'), 1000);
            }
        }
        
        // Handle Enter key
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Focus on input
        document.getElementById('messageInput').focus();
        
        // Ensure chat is scrolled to bottom on load
        window.addEventListener('load', function() {
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.scrollTop = chatMessages.scrollHeight;
            loadTheme(); // Load saved theme
        });
        
        // Add some interactive effects
        document.querySelectorAll('.quick-action').forEach(button => {
            button.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-2px)';
            });
            button.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0)';
            });
        });
    </script>
</body>
</html>
'''

# --------- Knowledge Base Management ---------
def create_default_knowledge_base():
    """Create a comprehensive knowledge base about AI/ML"""
    knowledge_content = """
ARTIFICIAL INTELLIGENCE
Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that normally require human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding. AI systems can analyze data, make decisions, and learn from experience. The field began in the 1950s and has gone through several "AI winters" and "AI springs" of progress.

MACHINE LEARNING
Machine Learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Instead of writing specific rules, we provide examples and let the computer discover patterns on its own. Machine learning algorithms improve their performance as they are exposed to more data over time. There are three main types: supervised learning, unsupervised learning, and reinforcement learning.

DEEP LEARNING
Deep Learning is a type of machine learning that uses neural networks with many layers. These multiple layers allow the network to learn complex patterns from large amounts of data. Deep learning has revolutionized fields like computer vision, speech recognition, and natural language processing. It's what powers technologies like self-driving cars, advanced medical diagnosis, and intelligent personal assistants.

NEURAL NETWORKS
Neural Networks are computing systems inspired by the human brain. They consist of interconnected nodes called neurons organized in layers. Each connection has a weight that adjusts during learning. Neural networks excel at finding complex patterns in data and can approximate any function given enough data and training. They're the foundation of most modern AI systems.

CONVOLUTIONAL NEURAL NETWORKS
Convolutional Neural Networks (CNNs) are specialized neural networks for processing grid-like data such as images. CNNs use convolutional layers to automatically detect important features like edges, shapes, and patterns. They are widely used in computer vision applications including image recognition, object detection, and medical image analysis. CNNs have revolutionized how computers "see" and understand visual information.

NATURAL LANGUAGE PROCESSING
Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. NLP combines computational linguistics with machine learning to process and analyze large amounts of natural language data. Applications include machine translation, sentiment analysis, chatbots, and text summarization. Modern NLP uses transformer architectures that have dramatically improved language understanding.

COMPUTER VISION
Computer Vision enables machines to interpret and understand visual information from the world. It involves methods for acquiring, processing, analyzing, and understanding digital images to produce numerical or symbolic information. Computer vision is used in facial recognition, medical image analysis, surveillance, and autonomous vehicles. It's one of the most rapidly advancing areas of AI.

REINFORCEMENT LEARNING
Reinforcement Learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. The agent learns through trial and error, receiving rewards or penalties for its actions. Reinforcement learning is used in game playing AI, robotics, and resource management. It's how AI systems learn complex behaviors through experience.

AI ETHICS
AI Ethics deals with the moral principles and guidelines for developing and using artificial intelligence responsibly. Key issues include algorithmic bias, fairness, transparency, privacy, accountability, and the societal impact of AI systems. Ethical AI ensures that technologies benefit humanity while minimizing potential harms. This includes addressing bias in training data and ensuring AI decisions are explainable.

TRANSFORMER ARCHITECTURE
Transformer architecture is a neural network design that uses self-attention mechanisms to process sequential data. Transformers have revolutionized natural language processing and are the foundation for models like GPT and BERT. They excel at understanding context and relationships in data. The attention mechanism allows the model to focus on different parts of the input when processing each element.

GENERATIVE AI
Generative AI refers to artificial intelligence systems that can create new content such as text, images, music, or code. These systems learn patterns from existing data and generate novel outputs that resemble the training data. Examples include ChatGPT for text generation and DALL-E for image creation. Generative AI has opened up new possibilities for creative applications and content generation.

SUPERVISED LEARNING
Supervised Learning is a machine learning approach where the model is trained on labeled data. Each training example is paired with an output label, and the model learns to map inputs to outputs. Common applications include classification and regression problems. Supervised learning requires large amounts of labeled data but typically produces highly accurate models.

UNSUPERVISED LEARNING
Unsupervised Learning involves training models on data without labeled responses. The system tries to learn the patterns and structure from the input data alone. Common techniques include clustering and dimensionality reduction. Unsupervised learning is useful for discovering hidden patterns in data and is often used for exploratory data analysis.

BIAS IN AI
Bias in AI occurs when algorithms produce systematically prejudiced results due to biased training data or flawed assumptions. This can lead to unfair treatment of certain groups and is a major concern in AI ethics and fairness. Addressing bias requires careful data collection, algorithm design, and ongoing monitoring of AI systems in production.

EXPLAINABLE AI
Explainable AI refers to methods and techniques that make the decisions and actions of AI systems understandable to humans. This is crucial for building trust, ensuring fairness, and meeting regulatory requirements in critical applications. Explainable AI helps users understand why an AI system made a particular decision, which is especially important in healthcare, finance, and criminal justice.
"""
    try:
        with open(KNOWLEDGE_BASE_PATH, 'w', encoding='utf-8') as f:
            f.write(knowledge_content.strip())
        print("📝 Created comprehensive knowledge base")
        return knowledge_content
    except Exception as e:
        print(f"❌ Error creating knowledge base: {e}")
        return ""

def load_knowledge_base():
    """Load the knowledge base from file"""
    try:
        if os.path.exists(KNOWLEDGE_BASE_PATH):
            with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    sections = [s for s in content.split('\n\n') if s.strip()]
                    print(f"✅ Knowledge base loaded with {len(sections)} topics")
                    return content
        # Create default if doesn't exist or is empty
        return create_default_knowledge_base()
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")
        return create_default_knowledge_base()

# --------- Interactive Features ---------
class LearningCompanion:
    def __init__(self):
        self.user_progress = {
            'topics_explored': set(),
            'quizzes_taken': 0,
            'challenges_completed': 0,
            'learning_path': []
        }
        self.conversation_context = []
        
    def track_interaction(self, topic, question_type):
        """Track user interactions for personalized experience"""
        self.user_progress['topics_explored'].add(topic)
        self.conversation_context.append({
            'topic': topic,
            'type': question_type,
            'timestamp': datetime.now()
        })
        
        # Keep only last 10 interactions
        if len(self.conversation_context) > 10:
            self.conversation_context.pop(0)
    
    def get_personalized_greeting(self):
        """Generate personalized greeting based on user progress"""
        topics_count = len(self.user_progress['topics_explored'])
        
        if topics_count == 0:
            return "🎉 Welcome! I'm excited to be your AI learning companion!"
        elif topics_count < 3:
            return f"👋 Welcome back! You've explored {topics_count} topics so far. Ready to learn more?"
        else:
            return f"🌟 Great to see you again! You're becoming quite the AI expert with {topics_count} topics explored!"
    
    def generate_learning_path_suggestion(self):
        """Suggest next learning steps based on user interests"""
        explored = self.user_progress['topics_explored']
        
        if not explored:
            return "Beginner", ["What is AI?", "Machine Learning Basics", "Real-world AI Applications"]
        
        if 'MACHINE LEARNING' in explored and 'DEEP LEARNING' not in explored:
            return "Intermediate", ["Deep Learning Fundamentals", "Neural Networks", "AI Model Training"]
        
        if 'DEEP LEARNING' in explored:
            return "Advanced", ["Transformer Architecture", "Generative AI", "AI Ethics and Challenges"]
        
        return "Continuing", ["Computer Vision", "Natural Language Processing", "Reinforcement Learning"]

# --------- Interactive Quizzes and Challenges ---------
AI_QUIZZES = {
    'beginner': [
        {
            'question': "What is the main goal of Machine Learning?",
            'options': [
                "To program computers with explicit rules",
                "To enable computers to learn from data", 
                "To replace human intelligence completely",
                "To create robot assistants"
            ],
            'correct': 1,
            'explanation': "Machine Learning focuses on enabling computers to learn patterns from data rather than being explicitly programmed with rules."
        }
    ],
    'intermediate': [
        {
            'question': "What makes Deep Learning 'deep'?",
            'options': [
                "It requires deep mathematical knowledge",
                "It uses neural networks with many layers",
                "It can solve deeply complex problems", 
                "It was developed by DeepMind"
            ],
            'correct': 1,
            'explanation': "Deep Learning is called 'deep' because it uses neural networks with multiple layers that can learn hierarchical representations of data."
        }
    ]
}

INTERACTIVE_CHALLENGES = {
    'MACHINE LEARNING': {
        'title': "🔄 ML Pattern Recognition Challenge",
        'description': "Can you identify which real-world problem is best solved with Machine Learning?",
        'problem': "Which scenario would benefit most from ML?",
        'options': [
            "Calculating 2+2",
            "Predicting house prices based on historical data",
            "Sorting a list of names alphabetically", 
            "Converting Celsius to Fahrenheit"
        ],
        'correct': 1,
        'explanation': "ML excels at finding patterns in historical data to make predictions, like estimating house prices!"
    }
}

# --------- Enhanced Response Generation ---------
def create_interactive_quiz(level='beginner'):
    """Generate an interactive quiz"""
    quiz = random.choice(AI_QUIZZES.get(level, AI_QUIZZES['beginner']))
    
    quiz_html = f'''
    <div class="challenge-box">
        <h3>🎯 Quick Knowledge Check!</h3>
        <p><strong>{quiz['question']}</strong></p>
        <div class="interactive-buttons">
            {"".join([f'<button class="interactive-btn" onclick="checkAnswer({i}, {quiz["correct"]}, `{quiz["explanation"]}`)">{option}</button>' 
                     for i, option in enumerate(quiz['options'])])}
        </div>
        <div id="quizResult" style="margin-top: 15px;"></div>
    </div>
    '''
    return quiz_html

def create_learning_path(companion):
    """Create personalized learning path"""
    level, suggestions = companion.generate_learning_path_suggestion()
    
    path_html = f'''
    <div class="learning-path">
        <h3>📚 Your Personalized Learning Path</h3>
        <p><strong>Current Level:</strong> {level}</p>
        <div class="progress-tracker">
            <strong>Suggested Next Steps:</strong><br>
            {" • ".join(suggestions)}
        </div>
        <div class="interactive-buttons">
            <button class="interactive-btn" onclick="askQuestion('{suggestions[0]}')">Start Next Topic</button>
            <button class="interactive-btn" onclick="askQuestion('quiz')">Test My Knowledge</button>
        </div>
    </div>
    '''
    return path_html

def create_interactive_challenge(topic):
    """Create an interactive challenge"""
    challenge = INTERACTIVE_CHALLENGES.get(topic, INTERACTIVE_CHALLENGES['MACHINE LEARNING'])
    
    challenge_html = f'''
    <div class="challenge-box">
        <h3>{challenge["title"]}</h3>
        <p>{challenge["description"]}</p>
        <p><strong>{challenge["problem"]}</strong></p>
        <div class="interactive-buttons">
            {"".join([f'<button class="challenge-btn" onclick="submitChallenge({i}, {challenge["correct"]}, `{challenge["explanation"]}`)">Option {i+1}</button>' 
                     for i in range(len(challenge['options']))])}
        </div>
        <div id="challengeResult"></div>
    </div>
    '''
    return challenge_html

# --------- Core AI Functions (Enhanced) ---------
companion = LearningCompanion()

def find_relevant_topic(question, knowledge_content):
    """Find the most relevant topic for the question"""
    question_lower = question.lower().strip()
    
    # Enhanced topic mapping with more keywords
    topic_keywords = {
        'ARTIFICIAL INTELLIGENCE': ['ai', 'artificial intelligence', 'what is ai', 'explain ai', 'define ai', 'intelligence'],
        'MACHINE LEARNING': ['machine learning', 'ml', 'what is ml', 'what is machine learning', 'explain ml', 'supervised', 'unsupervised'],
        'DEEP LEARNING': ['deep learning', 'deep neural', 'what is deep learning', 'multiple layers'],
        'NEURAL NETWORKS': ['neural network', 'neural networks', 'what is neural', 'explain neural', 'neurons'],
        'CONVOLUTIONAL NEURAL NETWORKS': ['cnn', 'convolutional', 'convolutional neural', 'what is cnn', 'image recognition'],
        'NATURAL LANGUAGE PROCESSING': ['nlp', 'natural language', 'language processing', 'what is nlp', 'text processing'],
        'COMPUTER VISION': ['computer vision', 'vision', 'image recognition', 'what is computer vision', 'visual recognition'],
        'REINFORCEMENT LEARNING': ['reinforcement learning', 'reinforcement', 'q-learning', 'reward learning'],
        'AI ETHICS': ['ai ethics', 'ethics', 'bias', 'fairness', 'ethical ai', 'ai bias', 'responsible ai'],
        'GENERATIVE AI': ['generative ai', 'gpt', 'chatgpt', 'dall-e', 'generative', 'create ai'],
        'TRANSFORMER ARCHITECTURE': ['transformer', 'attention', 'bert', 'gpt architecture', 'self-attention'],
        'BIAS IN AI': ['bias in ai', 'ai bias', 'algorithmic bias', 'unfair ai'],
        'EXPLAINABLE AI': ['explainable ai', 'interpretable ai', 'ai transparency', 'understandable ai']
    }
    
    # Special interactive commands
    if any(cmd in question_lower for cmd in ['quiz', 'test', 'challenge']):
        return 'QUIZ', 10
    if any(cmd in question_lower for cmd in ['learning path', 'progress', 'what next']):
        return 'LEARNING_PATH', 10
    if any(cmd in question_lower for cmd in ['help', 'what can you do']):
        return 'HELP', 10
    
    # Calculate scores for each topic
    topic_scores = {}
    for topic, keywords in topic_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in question_lower:
                # Exact matches get higher scores
                if question_lower == keyword or f" {keyword} " in f" {question_lower} ":
                    score += 3
                else:
                    score += 1
        if score > 0:
            topic_scores[topic] = score
    
    if topic_scores:
        # Return the topic with highest score
        best_topic = max(topic_scores.items(), key=lambda x: x[1])
        return best_topic[0], best_topic[1]
    
    return None, 0

def extract_answer(question, topic, knowledge_content):
    """Extract answer from knowledge base using QA model"""
    try:
        # Find the section for the topic
        sections = knowledge_content.split('\n\n')
        topic_section = None
        
        for section in sections:
            if section.startswith(topic):
                topic_section = section
                break
        
        if not topic_section:
            return None, 0
        
        # Use QA model to extract answer
        result = qa_pipeline(
            question=question,
            context=topic_section,
            max_answer_len=200,
            handle_impossible_answer=True
        )
        
        if result['score'] > 0.1 and result['answer']:
            return result['answer'], result['score']
        
        # Fallback: return the first meaningful part of the section
        lines = topic_section.split('\n')
        if len(lines) > 1:
            content = ' '.join(lines[1:])  # Skip the title line
            # Get first sentence or first 200 characters
            first_sentence = re.split(r'[.!?]', content)[0]
            if first_sentence and len(first_sentence.strip()) > 10:
                return first_sentence.strip() + '.', 0.3
            else:
                return content[:200] + '...', 0.2
                
    except Exception as e:
        print(f"Error extracting answer: {e}")
    
    return None, 0

def get_engaging_analogy(topic):
    """Get engaging analogies for topics"""
    analogies = {
        'ARTIFICIAL INTELLIGENCE': "🤖 Imagine AI as building a robot brain that can learn and think like humans, but potentially faster and for very specific tasks!",
        'MACHINE LEARNING': "🍎 Think of ML like teaching a child to recognize fruits - you show many examples, and soon they can identify new fruits they've never seen!",
        'DEEP LEARNING': "🧠 Deep Learning is like having a team of experts where each expert looks for specific patterns, and they combine their knowledge to understand complex things!",
        'CONVOLUTIONAL NEURAL NETWORKS': "👁️ CNNs are like giving computers super-powered eyes that can automatically detect edges, shapes, and objects in images!",
        'AI ETHICS': "⚖️ AI Ethics is like having traffic rules for self-driving cars - without proper guidelines, AI could cause harm instead of helping society!",
        'GENERATIVE AI': "🎨 Generative AI is like having a creative partner that can help you write stories, create art, or compose music based on patterns it has learned!"
    }
    return analogies.get(topic, "🚀 Think of this as technology that helps computers learn and make smart decisions, making our lives easier and more efficient!")

def get_exciting_examples(topic):
    """Get exciting real-world examples"""
    examples = {
        'ARTIFICIAL INTELLIGENCE': "🌟 AI powers amazing technologies like: Self-driving cars navigating complex roads, Virtual assistants understanding your voice, Medical AI detecting diseases early, and Netflix recommending your next favorite show!",
        'MACHINE LEARNING': "💫 ML is everywhere: Gmail filtering spam automatically, Banks detecting fraudulent transactions, Weather apps predicting storms days in advance, and Amazon suggesting products you'll love!",
        'DEEP LEARNING': "🔥 Deep Learning enables: Facebook recognizing your friends in photos, Voice assistants understanding natural speech, Medical systems analyzing X-rays with expert accuracy, and Self-driving cars seeing and understanding their environment!",
        'CONVOLUTIONAL NEURAL NETWORKS': "📸 CNNs power: Your phone's face unlock feature, Instagram filters that transform images, Security systems detecting intruders, and Medical imaging finding tiny abnormalities!",
        'AI ETHICS': "🛡️ Ethical AI ensures: Hiring algorithms are fair to all candidates, Facial recognition works equally well for all skin tones, AI systems protect your privacy, and Technology benefits everyone in society!"
    }
    return examples.get(topic, "💡 This technology is used in countless applications that make our world smarter, safer, and more efficient every day!")

def get_interactive_key_points(topic):
    """Get interactive key points"""
    points = {
        'ARTIFICIAL INTELLIGENCE': [
            "🤖 Creates intelligent systems that can learn and adapt",
            "💡 Powers technologies from voice assistants to self-driving cars", 
            "🌍 Transforming industries and creating new possibilities",
            "🚀 One of the most exciting fields in technology today!"
        ],
        'MACHINE LEARNING': [
            "📊 Learns patterns from data automatically",
            "🎯 Gets smarter with more examples and experience",
            "⚡ Can process information faster than humans",
            "💼 Used in finance, healthcare, entertainment, and more!"
        ],
        'DEEP LEARNING': [
            "🧠 Uses multi-layer neural networks for complex tasks",
            "👁️ Excellent for images, speech, and language understanding",
            "📈 Performance improves dramatically with more data",
            "🎨 Powers creative AI like image generation and music composition"
        ]
    }
    return points.get(topic, [
        "🚀 Helps solve complex problems automatically",
        "💡 Makes technology more intelligent and responsive", 
        "🌍 Used in applications that impact millions of people",
        "🎯 Continuously learning and improving over time"
    ])

def get_motivational_fact():
    """Get motivational facts about AI"""
    facts = [
        "💫 The AI market is growing exponentially - learning AI skills today could open amazing career opportunities tomorrow!",
        "🚀 Many of the world's most valuable companies are AI-first companies - your AI knowledge could be your superpower!",
        "🌍 AI is solving some of humanity's biggest challenges, from climate change to disease diagnosis!",
        "🎯 The AI you're learning about today will shape the technology of tomorrow - you're learning the future!",
        "💡 Many groundbreaking AI discoveries were made by people who started just like you - curious and eager to learn!",
        "🌟 The field of AI is less than 70 years old, yet it's already transforming our world - imagine what's next!",
        "🎨 AI is not just about technology - it's combining with art, music, and creativity in amazing ways!",
        "🔄 The AI revolution is compared to the industrial revolution in its potential impact - and you're part of it!"
    ]
    return random.choice(facts)

def create_progress_tracker(companion):
    """Create a progress tracker"""
    topics_explored = len(companion.user_progress['topics_explored'])
    
    progress_html = f'''
    <div class="progress-tracker">
        <h3>📊 Your Learning Progress</h3>
        <p><strong>Topics Explored:</strong> {topics_explored}</p>
        <p><strong>Quizzes Taken:</strong> {companion.user_progress['quizzes_taken']}</p>
        <p><strong>Challenges Completed:</strong> {companion.user_progress['challenges_completed']}</p>
        <div style="background: linear-gradient(90deg, #3498db {min(topics_explored*10, 100)}%, #ecf0f1 {min(topics_explored*10, 100)}%); 
                    height: 20px; border-radius: 10px; margin: 10px 0;"></div>
        <p>Keep going! Every topic you explore brings you closer to AI mastery! 🚀</p>
    </div>
    '''
    return progress_html

# --------- Enhanced Question Processing ---------
def generate_impressive_response(question, knowledge_content, user_level='beginner', message_count=0):
    """Generate impressive, interactive responses"""
    
    # Handle special interactive commands first
    question_lower = question.lower().strip()
    
    if any(cmd in question_lower for cmd in ['hi', 'hello', 'hey', 'greetings']):
        greeting = companion.get_personalized_greeting()
        return f'''
        <div class="info-card">
            <h3>{greeting}</h3>
            <p>I'm your AI Learning Companion, here to make your journey into artificial intelligence exciting and engaging!</p>
            {create_progress_tracker(companion)}
            <div class="interactive-buttons">
                <button class="interactive-btn" onclick="askQuestion('What is AI?')">Start Learning</button>
                <button class="interactive-btn" onclick="askQuestion('Show me cool AI applications')">See Amazing AI</button>
                <button class="interactive-btn" onclick="askQuestion('Take a quiz')">Test My Knowledge</button>
            </div>
        </div>
        '''
    
    if any(cmd in question_lower for cmd in ['quiz', 'test', 'challenge']):
        companion.user_progress['quizzes_taken'] += 1
        return create_interactive_quiz(user_level)
    
    if any(cmd in question_lower for cmd in ['learning path', 'progress', 'what should i learn']):
        return create_learning_path(companion)
    
    if any(cmd in question_lower for cmd in ['help', 'what can you do']):
        return '''
        <div class="info-card">
            <h3>🎯 How I Can Help You Learn AI</h3>
            <p><strong>Here's what we can do together:</strong></p>
            <div class="key-points-card">
                • <strong>Explain Concepts</strong>: Ask about any AI topic!<br>
                • <strong>Interactive Quizzes</strong>: Test your knowledge<br>
                • <strong>Learning Path</strong>: Personalized guidance<br>
                • <strong>Real Examples</strong>: See AI in action<br>
                • <strong>Challenges</strong>: Hands-on learning<br>
                • <strong>Progress Tracking</strong>: See how you're growing!
            </div>
            <p>Try asking: "What is machine learning?" or "Show me AI applications" or "Give me a quiz"!</p>
        </div>
        '''
    
    # Find relevant topic
    topic, topic_score = find_relevant_topic(question, knowledge_content)
    
    if not topic or topic_score == 0:
        return '''
        <div class="unrelated-warning">
            <h3>🎯 Let's Explore AI Together!</h3>
            <p>I specialize in making Artificial Intelligence and Machine Learning concepts fun and easy to understand!</p>
            <div class="interactive-buttons">
                <button class="interactive-btn" onclick="askQuestion('What is AI?')">AI Basics</button>
                <button class="interactive-btn" onclick="askQuestion('Machine Learning examples')">ML Examples</button>
                <button class="interactive-btn" onclick="askQuestion('Take a quiz')">Quick Quiz</button>
                <button class="interactive-btn" onclick="askQuestion('Learning path')">My Path</button>
            </div>
        </div>
        '''
    
    # Track user interaction
    companion.track_interaction(topic, 'question')
    
    # Extract answer from knowledge base
    answer, confidence = extract_answer(question, topic, knowledge_content)
    
    if not answer:
        answer = f"{topic} represents one of the most exciting areas in technology today, helping computers solve complex problems and learn from experience!"
    
    # Generate impressive response
    response = f'''
    <div class="answer-box">
        <div class="topic-badge">{topic}</div>
        
        <div class="info-card">
            <h3>📚 Here's What You Asked About:</h3>
            <p>{answer}</p>
        </div>
        
        <div class="info-card">
            <h3>💡 Making It Simple:</h3>
            <p>{get_engaging_analogy(topic)}</p>
        </div>
        
        <div class="key-points-card">
            <h3>🎯 Key Insights:</h3>
            {"".join([f'• {point}<br>' for point in get_interactive_key_points(topic)])}
        </div>
        
        <div class="example-card">
            <h3>🌍 Real-World Impact:</h3>
            <p>{get_exciting_examples(topic)}</p>
        </div>
        
        <div class="fun-fact-card">
            <h3>🌟 Motivational Moment:</h3>
            <p>{get_motivational_fact()}</p>
        </div>
        
        {create_progress_tracker(companion)}
        
        <div class="interactive-buttons">
            <button class="interactive-btn" onclick="askQuestion('quiz')">Test My Knowledge 🎯</button>
            <button class="interactive-btn" onclick="askQuestion('What is next?')">Continue Learning 📚</button>
            <button class="interactive-btn" onclick="askQuestion('More about {topic}')">Dive Deeper 🔍</button>
        </div>
    </div>
    '''
    
    return response

# --------- Load Knowledge Base ---------
KNOWLEDGE_CONTENT = load_knowledge_base()

# --------- Flask Routes ---------
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        message_count = data.get('message_count', 0)
        user_level = data.get('user_level', 'beginner')
        
        if not question:
            return jsonify({'success': False, 'answer': 'Please ask a question.'})
        
        print(f"💭 Question: {question}")
        
        # Generate impressive answer
        answer = generate_impressive_response(question, KNOWLEDGE_CONTENT, user_level, message_count)
        
        return jsonify({
            'success': True,
            'answer': answer
        })
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({
            'success': False,
            'answer': '''
            <div class="unrelated-warning">
                <h3>😅 Oops! Let's try that again</h3>
                <p>I encountered a small issue. How about we explore something amazing about AI instead?</p>
                <div class="interactive-buttons">
                    <button class="interactive-btn" onclick="askQuestion('What is AI?')">Start Fresh</button>
                    <button class="interactive-btn" onclick="askQuestion('Take a quiz')">Quick Quiz</button>
                </div>
            </div>
            '''
        })

@app.route('/health')
def health():
    sections = [s for s in KNOWLEDGE_CONTENT.split('\n\n') if s.strip()]
    topics = [s.split('\n')[0] for s in sections if s.split('\n')[0]]
    
    return jsonify({
        'status': 'healthy',
        'topics_loaded': len(topics),
        'user_progress': companion.user_progress,
        'model': MODEL_NAME
    })

# --------- Startup ---------
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🤖 AI Learning Companion - Interactive Education Platform")
    print("="*70)
    print(f"📚 Knowledge Base: {len([s for s in KNOWLEDGE_CONTENT.split('\\n\\n') if s.strip()])} topics")
    print(f"🧠 AI Model: {MODEL_NAME}")
    print("✨ Enhanced Features:")
    print("• 🎯 Interactive quizzes and challenges")
    print("• 📊 Personalized learning progress tracking") 
    print("• 🚀 Engaging animations and visual design")
    print("• 💡 Motivational learning experience")
    print("• 🌟 Real-world examples and applications")
    print("• 🎨 Beautiful gradient designs and interactive elements")
    print("="*70)
    print("🌐 Starting server at http://localhost:5000")
    print("="*70)
    
    app.run(host='0.0.0.0', port=5000, debug=False)