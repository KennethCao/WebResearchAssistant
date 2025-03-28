// 使用严格模式
'use strict';

// 使用立即执行函数避免全局变量污染
(function() {
    // 问答功能
    async function sendQuestion(question) {
        try {
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
        } catch (error) {
            console.error('Error:', error);
            throw error;
        }
    }

    // 刷新分析结果
    async function refreshAnalysis() {
        const refreshBtn = document.getElementById('refresh-analysis');
        if (!refreshBtn) return;
        
        refreshBtn.disabled = true;
        refreshBtn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> 刷新中...';
        
        try {
            const response = await fetch('/api/analysis/refresh', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('刷新失败');
            }
            
            location.reload();
        } catch (error) {
            console.error('刷新分析结果失败:', error);
            alert('刷新分析结果失败，请稍后重试');
        } finally {
            refreshBtn.disabled = false;
            refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> 刷新分析';
        }
    }

    // 添加事件监听器
    document.addEventListener('DOMContentLoaded', function() {
        const refreshBtn = document.getElementById('refresh-analysis');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', refreshAnalysis);
        }
    });
})(); 