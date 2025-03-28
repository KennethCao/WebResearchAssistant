'use strict';

class UploadHandler {
    constructor() {
        this.form = document.getElementById('uploadForm');
        this.progress = document.getElementById('uploadProgress');
        this.progressBar = this.progress.querySelector('.progress-bar');
        this.progressText = document.getElementById('progressText');
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.handleUpload();
        });
    }

    async handleUpload() {
        const formData = new FormData(this.form);
        
        try {
            this.showProgress();
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Upload failed');
            }
            
            const data = await response.json();
            
            if (data.success) {
                window.app.showSuccess('File uploaded successfully');
                window.location.href = `/analysis/${data.paper_id}`;
            } else {
                throw new Error(data.error);
            }
            
        } catch (error) {
            window.app.showError('Upload failed: ' + error.message);
        } finally {
            this.hideProgress();
        }
    }

    showProgress() {
        this.progress.classList.remove('d-none');
        this.progressBar.style.width = '0%';
        this.progressText.textContent = 'Uploading...';
    }

    hideProgress() {
        this.progress.classList.add('d-none');
    }

    updateProgress(percent) {
        this.progressBar.style.width = `${percent}%`;
        this.progressText.textContent = `Uploading... ${percent}%`;
    }
}

class PapersList {
    constructor() {
        this.tableBody = document.getElementById('papersList');
        this.refreshBtn = document.getElementById('refreshList');
        this.setupEventListeners();
        this.loadPapers();
    }

    setupEventListeners() {
        if (this.refreshBtn) {
            this.refreshBtn.addEventListener('click', () => this.loadPapers());
        }
    }

    async loadPapers() {
        try {
            const response = await fetch('/papers');
            const papers = await response.json();
            this.renderPapers(papers);
        } catch (error) {
            window.app.showError('Failed to load papers');
        }
    }

    renderPapers(papers) {
        this.tableBody.innerHTML = papers.map(paper => `
            <tr>
                <td>${paper.title || paper.filename}</td>
                <td>${new Date(paper.uploaded_at).toLocaleString()}</td>
                <td>
                    <span class="badge bg-${this.getStatusBadge(paper.analysis_status)}">
                        ${paper.analysis_status}
                    </span>
                </td>
                <td>
                    <a href="/analysis/${paper.id}" class="btn btn-sm btn-primary">
                        <i class="fas fa-eye"></i> View
                    </a>
                </td>
            </tr>
        `).join('');
    }

    getStatusBadge(status) {
        switch (status) {
            case 'completed': return 'success';
            case 'pending': return 'warning';
            case 'failed': return 'danger';
            default: return 'secondary';
        }
    }
}

// Initialize components when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new UploadHandler();
    new PapersList();
}); 