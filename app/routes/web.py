from flask import Blueprint, render_template, request, jsonify, send_file, flash, redirect, url_for, abort, current_app, send_from_directory
from app.services.paper_service import PaperService
from app.utils.logger import setup_logger
from werkzeug.utils import secure_filename
import os
from app.nlp.blockchain_qa import BlockchainQA
from app.utils.analysis import get_analysis_status
from app.utils.keywords import extract_keywords
from app.models.paper import Paper
from app.extensions import db
from sqlalchemy.exc import OperationalError
from app.nlp.analyzer import analyze_paper
import threading
from collections import Counter
from datetime import datetime

bp = Blueprint('web', __name__)
logger = setup_logger(__name__)
paper_service = PaperService()

@bp.route('/')
def index():
    """首页"""
    try:
        # 获取最近上传的文件
        papers = Paper.query.order_by(Paper.uploaded_at.desc()).limit(5).all()
        return render_template('index.html', recent_papers=papers)
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return render_template('index.html', recent_papers=[])

@bp.route('/upload')
def upload():
    """上传页面"""
    return render_template('upload.html')

@bp.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    try:
        logger.info("Starting file upload process")
        
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if not (file.filename.lower().endswith('.pdf') and file.mimetype == 'application/pdf'):
            logger.warning(f"Invalid file type: {file.filename} ({file.mimetype})")
            return jsonify({'error': 'Only PDF files are allowed'}), 400
            
        if file:
            try:
                # 获取安全的文件名
                filename = secure_filename(file.filename)
                logger.info(f"Processing file: {filename}")
                
                # 获取上传目录路径（修复路径构造）
                upload_dir = os.path.join(
                    current_app.config['DATA_DIR'],  # 使用应用配置的目录
                    'uploads'
                )
                
                # 确保上传目录存在（增加权限检查）
                try:
                    os.makedirs(upload_dir, exist_ok=True)
                    os.chmod(upload_dir, 0o755)  # 确保目录可写
                    logger.info(f"Verified upload directory: {upload_dir}")
                except Exception as e:
                    logger.error(f"Directory creation failed: {e}")
                    return jsonify({'error': 'Failed to create upload directory'}), 500
                
                # 使用数据库事务
                try:
                    db.session.begin()
                    # 创建新的 Paper 记录
                    paper = Paper(
                        filename=filename,
                        file_path=os.path.relpath(file_path, start=current_app.config['DATA_DIR']),  # 存储相对路径
                        uploaded_at=datetime.now(),
                        analysis_status='pending'
                    )
                    db.session.add(paper)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    logger.error(f"Database transaction failed: {e}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return jsonify({'error': 'Database operation failed'}), 500
                
                logger.info(f"Upload directory: {upload_dir}")
                
                # 保存文件
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                logger.info(f"File saved to: {file_path}")
                # 验证PDF文件有效性
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(4)
                        if header != b'%PDF':
                            os.remove(file_path)
                            return jsonify({'error': 'Invalid PDF file format'}), 400
                except Exception as e:
                    logger.error(f"PDF validation failed: {e}")
                    os.remove(file_path)
                    return jsonify({'error': 'Failed to validate PDF file'}), 500
                
                try:
                    # 创建新的 Paper 记录
                    paper = Paper(
                        filename=filename,
                        file_path=file_path,
                        uploaded_at=datetime.now(),
                        analysis_status='pending'
                    )
                    
                    # 保存到数据库
                    db.session.add(paper)
                    db.session.commit()
                    logger.info(f"Created paper record with ID: {paper.id}")
                    
                    # 启动异步分析
                    thread = threading.Thread(
                        target=paper_service.process_paper,
                        args=(paper.id,)
                    )
                    thread.start()
                    
                    return jsonify({
                        'message': 'File uploaded successfully',
                        'paper_id': paper.id
                    }), 200
                    
                except Exception as e:
                    logger.error(f"Database error: {e}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return jsonify({'error': 'Failed to process upload'}), 500
                    
            except Exception as e:
                logger.error(f"File save error: {e}")
                return jsonify({'error': 'Failed to save file'}), 500
                
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Failed to process upload'}), 500

@bp.route('/papers')
def papers():
    """文献列表页面"""
    try:
        # 获取所有文献
        papers = Paper.query.order_by(Paper.uploaded_at.desc()).all()
        return render_template('papers.html', papers=papers)
    except Exception as e:
        logger.error(f"Error getting paper list: {e}")
        return render_template('papers.html', papers=[])

@bp.route('/papers/<int:paper_id>')
def view_paper(paper_id):
    """查看论文详情"""
    try:
        paper = Paper.query.get_or_404(paper_id)
        return render_template('paper_details.html', paper=paper)
    except Exception as e:
        logger.error(f"Error viewing paper {paper_id}: {e}")
        flash('Failed to load paper details', 'error')
        return redirect(url_for('web.papers'))

@bp.route('/papers/<int:paper_id>/result')
def view_analysis(paper_id):
    """查看分析结果"""
    try:
        paper = Paper.query.get_or_404(paper_id)
        if paper.analysis_status == 'processing':
            return render_template('analysis_waiting.html', paper=paper)
        return render_template('analysis.html', paper=paper)
    except Exception as e:
        logger.error(f"Error showing analysis results: {e}")
        flash('Failed to get analysis results', 'error')
        return redirect(url_for('web.index'))

@bp.route('/trends')
def trends():
    """趋势分析页面"""
    try:
        # 获取所有已分析完成的论文
        papers = Paper.query.filter_by(analysis_status='completed').all()
        
        # 获取趋势数据
        trend_data = {
            'total_papers': len(papers),
            'topics': {},
            'recent_papers': []
        }
        
        if papers:
            # 收集所有主题
            all_topics = []
            for paper in papers:
                if paper.topics:
                    topics = [t.strip() for t in paper.topics.split(',') if t.strip()]
                    all_topics.extend(topics)
            
            # 计算主题频率
            topic_counter = Counter(all_topics)
            
            # 计算主题权重 (用于标签云显示)
            if topic_counter:
                max_count = max(topic_counter.values())
                trend_data['topics'] = {
                    topic: 1 + (count / max_count) * 2
                    for topic, count in topic_counter.most_common(10)  # 只显示前10个主题
                }
            
            # 获取最近的论文（按上传时间排序）
            recent_papers = sorted(papers, key=lambda x: x.uploaded_at, reverse=True)[:10]
            for paper in recent_papers:
                trend_data['recent_papers'].append({
                    'id': paper.id,
                    'title': paper.title or paper.filename,
                    'keywords': paper.keywords,
                    'uploaded_at': paper.uploaded_at,
                    'summary': paper.summary[:200] + '...' if paper.summary else None
                })
        
        logger.info(f"Generated trend data with {trend_data['total_papers']} papers")
        return render_template('trends.html', trend_data=trend_data)
        
    except Exception as e:
        logger.error(f"Error getting trend data: {e}")
        return render_template('trends.html', trend_data={
            'total_papers': 0,
            'topics': {},
            'recent_papers': []
        })

@bp.route('/qa')
def qa():
    """问答页面"""
    try:
        # 获取知识库状态
        qa_system = BlockchainQA()
        kb_status = {
            'document_count': len(qa_system.knowledge_base) if qa_system.knowledge_base else 0
        }
        return render_template('qa.html', kb_status=kb_status)
    except Exception as e:
        logger.error(f"Error loading QA page: {e}")
        return render_template('qa.html', kb_status={'document_count': 0})

@bp.route('/papers/file/<filename>')
def show_paper_file(filename):
    """返回上传的文件"""
    try:
        # 获取上传目录路径
        upload_folder = os.path.join(
            os.getenv('BLOCKCHAIN_ASSISTANT_DATA_DIR'),
            'uploads'
        )
        
        # 检查文件是否存在
        if not os.path.exists(os.path.join(upload_folder, filename)):
            abort(404)
            
        # 返回文件
        return send_from_directory(upload_folder, filename)
    except Exception as e:
        current_app.logger.error(f"Error serving file {filename}: {e}")
        abort(500)

# 添加错误处理
@bp.errorhandler(404)
def not_found_error(error):
    """处理404错误"""
    logger.error(f"Page not found: {request.url}")
    return render_template('errors/404.html'), 404

@bp.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    logger.error(f"Server error: {error}")
    db.session.rollback()  # 回滚数据库会话
    return render_template('errors/500.html', error=str(error)), 500

def process_paper(paper_id):
    """异步处理上传的论文"""
    try:
        with current_app.app_context():
            # 获取论文
            paper = Paper.query.get(paper_id)
            if not paper:
                logger.error(f"Paper {paper_id} not found")
                return
                
            # 更新状态为处理中
            paper.analysis_status = 'processing'
            db.session.commit()
            
            try:
                # 分析论文
                analysis_result = analyze_paper(paper.file_path)
                
                # 更新论文信息
                paper.title = analysis_result.get('title')
                paper.keywords = analysis_result.get('keywords')
                paper.topics = analysis_result.get('topics')
                paper.summary = analysis_result.get('summary')
                paper.analysis_status = 'completed'
                
                # 保存到数据库
                db.session.commit()
                logger.info(f"Paper {paper_id} analysis completed")
                
            except Exception as e:
                logger.error(f"Failed to analyze paper {paper_id}: {e}")
                paper.analysis_status = 'failed'
                db.session.commit()
                
    except Exception as e:
        logger.error(f"Error in process_paper: {e}")
        file_path = os.path.join(current_app.config['DATA_DIR'], filename)