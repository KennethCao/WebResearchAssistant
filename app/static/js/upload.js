/* eslint-disable */
// @ts-check

// 等待 DOM 加载完成
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const status = document.getElementById('upload-status');
    
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            try {
                const fileInput = form.querySelector('input[type="file"]');
                if (!fileInput.files.length) {
                    showStatus('请选择文件', 'error');
                    return;
                }
                
                const file = fileInput.files[0];
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    showStatus('只支持 PDF 文件', 'error');
                    return;
                }
                
                if (file.size > 16 * 1024 * 1024) {
                    showStatus('文件大小不能超过 16MB', 'error');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                showStatus('正在上传...', 'info');
                
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || `Upload failed with status: ${response.status}`);
                }
                
                showStatus('上传成功！正在分析...', 'success');
                
                // 开始轮询分析状态
                if (data.paper_id) {
                    pollAnalysisStatus(data.paper_id);
                }
                
            } catch (error) {
                console.error('Upload error:', error);
                showStatus(error.message || '上传失败，请重试', 'error');
            }
        });
    }
    
    function showStatus(message, type) {
        if (status) {
            status.textContent = message;
            status.className = `alert alert-${type}`;
            status.style.display = 'block';
        }
    }
    
    async function pollAnalysisStatus(paperId) {
        try {
            const response = await fetch(`/api/papers/${paperId}/status`);
            const data = await response.json();
            
            if (data.status === 'completed') {
                showStatus('分析完成！', 'success');
                setTimeout(() => {
                    window.location.href = `/papers/${paperId}/result`;
                }, 1500);
            } else if (data.status === 'error') {
                showStatus(`分析失败: ${data.error}`, 'error');
            } else {
                showStatus('正在分析...', 'info');
                setTimeout(() => pollAnalysisStatus(paperId), 2000);
            }
        } catch (error) {
            showStatus('检查状态失败，请刷新页面', 'error');
        }
    }
});

// 检查知识库状态
async function checkKnowledgeBase() {
    try {
        const response = await fetch('/api/kb/status');
        const data = await response.json();
        
        if (data.status === 'success') {
            const kbStatus = document.getElementById('kb-status');
            if (kbStatus) {
                kbStatus.innerHTML = `Current knowledge base contains ${data.document_count} papers`;
            }
        }
    } catch (error) {
        console.error('Failed to check knowledge base:', error);
    }
}

// Called after file upload success
checkKnowledgeBase();

// 文件上传相关的 JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const selectFileBtn = document.getElementById('select-file');
    const uploadForm = document.getElementById('upload-form');
    const status = document.getElementById('upload-status');
    const progress = document.getElementById('analysis-progress');
    const progressBar = document.querySelector('.progress');
    const progressBarInner = document.querySelector('.progress-bar');

    // 防止浏览器默认行为
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // 处理拖放
    dropZone.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        
        if (file) {
            handleFileUpload(file);
        }
    }

    // 点击选择文件按钮时触发文件输入框点击
    selectFileBtn.addEventListener('click', function() {
        fileInput.click();
    });

    // 监听文件选择变化
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFileUpload(this.files[0]);
        }
    });

    // 处理文件上传
    function handleFileUpload(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        // 显示进度条
        progressBar.style.display = 'block';
        progressBarInner.style.width = '0%';
        
        // 重置状态显示
        status.style.display = 'none';
        progress.textContent = '';
        
        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || `HTTP error! status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            status.className = 'alert alert-success';
            status.style.display = 'block';
            status.textContent = data.message;
            
            // 开始轮询分析状态
            if (data.paper_id) {
                pollAnalysisStatus(data.paper_id);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            status.className = 'alert alert-danger';
            status.style.display = 'block';
            status.textContent = error.message || 'Upload failed';
            progressBar.style.display = 'none';
        });
    }

    // 轮询分析状态
    async function pollAnalysisStatus(paperId) {
        try {
            const response = await fetch(`/api/papers/${paperId}/status`);
            const data = await response.json();
            
            if (data.error) {
                showMessage(`分析失败: ${data.error}`, 'error');
                return;
            }
            
            if (data.status === 'completed') {
                showMessage('分析完成！', 'success');
                setTimeout(() => {
                    window.location.href = `/papers/${paperId}`;
                }, 1500);
            } else if (data.status === 'error') {
                showMessage(`分析失败: ${data.error}`, 'error');
            } else {
                showMessage(`${data.current_task}...`, 'info');
                setTimeout(() => pollAnalysisStatus(paperId), 2000);
            }
        } catch (error) {
            showMessage('检查状态失败，请刷新页面', 'error');
        }
    }
});

async function uploadFile(formData) {
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        return data;
    } catch (error) {
        console.error('Upload error:', error);
        throw error;
    }
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('upload-status');
    statusDiv.className = `alert alert-${type}`;
    statusDiv.textContent = message;
    statusDiv.style.display = 'block';
} 