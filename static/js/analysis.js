'use strict';

class AnalysisPage {
    constructor() {
        this.refreshBtn = document.getElementById('refreshAnalysis');
        this.paperId = this.getPaperId();
        this.setupEventListeners();
    }

    getPaperId() {
        const path = window.location.pathname;
        return path.split('/').pop();
    }

    setupEventListeners() {
        if (this.refreshBtn) {
            this.refreshBtn.addEventListener('click', () => this.refreshAnalysis());
        }
    }

    async refreshAnalysis() {
        try {
            this.refreshBtn.disabled = true;
            this.refreshBtn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> Refreshing...';

            const response = await fetch(`/api/papers/${this.paperId}/refresh`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Refresh failed');
            }

            window.location.reload();

        } catch (error) {
            window.app.showError('Failed to refresh analysis');
        } finally {
            this.refreshBtn.disabled = false;
            this.refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh Analysis';
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new AnalysisPage();
}); 