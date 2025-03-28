from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from app.utils.logger import setup_logger
from app.utils.analysis import get_analysis_status, start_analysis_task
from app.qa.blockchain_qa import BlockchainQA

web = Blueprint('web', __name__)
logger = setup_logger(__name__)

@web.route('/')
def index():
    """Home page with document upload and list"""
    try:
        # Get list of uploaded documents from analysis cache
        from app.utils.analysis import analysis_cache
        documents = []
        for file_id, data in analysis_cache.items():
            documents.append({
                'file_id': file_id,
                'filename': data.get('filename', 'Unknown'),
                'upload_time': data.get('upload_time', 'Unknown'),
                'status': data.get('status', 'unknown')
            })
        
        # Sort by upload time, newest first
        documents.sort(key=lambda x: x['upload_time'], reverse=True)
        
        return render_template('blockchain_upload.html', documents=documents)
    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        return render_template('blockchain_upload.html', documents=[], error="Error loading documents")

@web.route('/upload', methods=['POST'])
def upload():
    """Handle document upload"""
    try:
        if 'file' not in request.files:
            return render_template('blockchain_upload.html', error="No file selected")
            
        file = request.files['file']
        if file.filename == '':
            return render_template('blockchain_upload.html', error="No file selected")
            
        if file and file.filename.endswith('.pdf'):
            # Save file
            filename = secure_filename(file.filename)
            file_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, f"{file_id}_{filename}")
            file.save(file_path)
            
            # Start analysis
            start_analysis_task(file_path, file_id)
            
            # Redirect to analysis page
            return redirect(url_for('web.analysis', file_id=file_id))
        else:
            return render_template('blockchain_upload.html', error="Please upload a PDF file")
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return render_template('blockchain_upload.html', error="Upload failed")

@web.route('/analysis/<file_id>')
def analysis(file_id):
    """Document analysis page"""
    try:
        analysis_data = get_analysis_status(file_id)
        if not analysis_data:
            return redirect(url_for('web.index'))
            
        return render_template('analysis.html',
            file_id=file_id,
            filename=analysis_data.get('filename', ''),
            upload_time=analysis_data.get('upload_time', ''),
            file_size=analysis_data.get('file_size', ''),
            summary=analysis_data.get('summary', ''),
            keywords=analysis_data.get('keywords', [])
        )
    except Exception as e:
        logger.error(f"Error loading analysis page: {e}")
        return redirect(url_for('web.index'))

@web.route('/qa/<file_id>')
def qa(file_id):
    """Q&A page"""
    try:
        analysis_data = get_analysis_status(file_id)
        if not analysis_data or analysis_data.get('status') != 'completed':
            return redirect(url_for('web.analysis', file_id=file_id))
            
        return render_template('qa.html',
            file_id=file_id,
            filename=analysis_data.get('filename', ''),
            summary=analysis_data.get('summary', '')
        )
    except Exception as e:
        logger.error(f"Error loading Q&A page: {e}")
        return redirect(url_for('web.index')) 