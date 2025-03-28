'use strict';

class App {
    constructor() {
        this.setupEventListeners();
        this.setupAjaxHandlers();
    }

    setupEventListeners() {
        // Handle flash message dismissal
        document.querySelectorAll('.alert .btn-close').forEach(button => {
            button.addEventListener('click', e => {
                e.target.closest('.alert').remove();
            });
        });

        // Auto-hide flash messages after 5 seconds
        setTimeout(() => {
            document.querySelectorAll('.alert').forEach(alert => {
                alert.classList.remove('show');
            });
        }, 5000);
    }

    setupAjaxHandlers() {
        // Add CSRF token to all AJAX requests
        const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content;
        if (csrfToken) {
            $.ajaxSetup({
                beforeSend: function(xhr) {
                    xhr.setRequestHeader('X-CSRFToken', csrfToken);
                }
            });
        }

        // Global AJAX error handler
        $(document).ajaxError((event, jqXHR, settings, error) => {
            this.showError('An error occurred while processing your request. Please try again.');
            console.error('AJAX Error:', error);
        });
    }

    showError(message) {
        const alert = `
            <div class="alert alert-danger alert-dismissible fade show">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        $('.container').first().prepend(alert);
    }

    showSuccess(message) {
        const alert = `
            <div class="alert alert-success alert-dismissible fade show">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        $('.container').first().prepend(alert);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
}); 