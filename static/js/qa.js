'use strict';

class QASystem {
    constructor() {
        this.messageContainer = document.getElementById('chatMessages');
        this.questionForm = document.getElementById('questionForm');
        this.questionInput = document.getElementById('questionInput');
        this.paperId = this.getPaperId();
        this.setupEventListeners();
    }

    getPaperId() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('paper_id');
    }

    setupEventListeners() {
        // Handle question form submission
        this.questionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.handleQuestion();
        });

        // Handle example questions
        document.querySelectorAll('.example-question').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.questionInput.value = e.target.textContent;
                this.handleQuestion();
            });
        });
    }

    async handleQuestion() {
        const question = this.questionInput.value.trim();
        if (!question) return;

        // Add user question to chat
        this.addMessage(question, 'user');
        this.questionInput.value = '';

        try {
            // Show typing indicator
            const typingMessage = this.addTypingIndicator();

            // Send question to server
            const response = await fetch('/api/qa/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    paper_id: this.paperId
                })
            });

            // Remove typing indicator
            typingMessage.remove();

            if (!response.ok) {
                throw new Error('Failed to get answer');
            }

            const data = await response.json();
            
            if (data.success) {
                this.addMessage(data.answer, 'bot');
            } else {
                throw new Error(data.error);
            }

        } catch (error) {
            console.error('Error:', error);
            this.addMessage(
                'Sorry, I encountered an error while processing your question. Please try again.',
                'bot error'
            );
        }
    }

    addMessage(text, type) {
        const div = document.createElement('div');
        div.className = `message ${type}-message fade-in`;
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = text;
        
        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = new Date().toLocaleTimeString();
        
        div.appendChild(content);
        div.appendChild(time);
        
        this.messageContainer.appendChild(div);
        this.scrollToBottom();
        
        return div;
    }

    addTypingIndicator() {
        const div = document.createElement('div');
        div.className = 'message bot-message typing-indicator';
        div.innerHTML = `
            <div class="message-content">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        
        this.messageContainer.appendChild(div);
        this.scrollToBottom();
        
        return div;
    }

    scrollToBottom() {
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
}

// Initialize QA system when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.qaSystem = new QASystem();
}); 