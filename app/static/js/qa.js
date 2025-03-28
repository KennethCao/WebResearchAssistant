'use strict';

class QASystem {
    constructor() {
        this.messageContainer = document.getElementById('chatMessages');
        this.questionForm = document.getElementById('questionForm');
        this.questionInput = document.getElementById('questionInput');
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.questionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.handleQuestion();
        });
    }

    async handleQuestion() {
        const question = this.questionInput.value.trim();
        if (!question) return;

        // 添加用户问题到界面
        this.addMessage(question, 'user');
        this.questionInput.value = '';

        try {
            const response = await this.sendQuestion(question);
            this.addMessage(response.answer, 'assistant');
        } catch (error) {
            console.error('Error:', error);
            this.addMessage('抱歉，处理您的问题时出现错误。', 'system');
        }
    }

    async sendQuestion(question) {
        const response = await fetch('/api/blockchain/qa', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        return await response.json();
    }

    addMessage(text, type) {
        const div = document.createElement('div');
        div.className = `message ${type}`;
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = text;
        
        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = new Date().toLocaleTimeString();
        
        div.appendChild(content);
        div.appendChild(time);
        
        this.messageContainer.appendChild(div);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
}

// 初始化QA系统
document.addEventListener('DOMContentLoaded', () => {
    window.qaSystem = new QASystem();
}); 