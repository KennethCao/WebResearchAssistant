# api.py
from typing import Dict, Any, List, Tuple
from flask import Blueprint, request, jsonify, Response, current_app
from werkzeug.utils import secure_filename
import os
from app.utils.logger import setup_logger
from app.config.environment import (
    UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH,
    LOG_FILE,
    API_PREFIX
)
from app.file_processing.file_validator import validate_file, allowed_file
from app.nlp.summarizer import TextSummarizer
from app.nlp.embedding_generator import EmbeddingGenerator
from app.rag.retriever import DocumentRetriever
# from flask_swagger_ui import get_swaggerui_blueprint  # 暂时注释掉
from app.nlp.blockchain_qa import BlockchainQA
from app.utils.analysis import get_analysis_status
from app.services.paper_service import PaperService
from app.services.qa_service import QAService
from PyPDF2 import PdfReader
import uuid
from threading import Thread
from app.models.paper import Paper
from app.extensions import db
from datetime import datetime
from app.nlp.analyzer import analyze_paper
import threading

# 设置日志记录器
logger = setup_logger(__name__)

# 创建蓝图，注意这里不要加 url_prefix
bp = Blueprint('api', __name__, url_prefix='/api')

# 初始化组件
summarizer = TextSummarizer()
embedding_generator = EmbeddingGenerator()
document_retriever = DocumentRetriever()

# 暂时注释掉 swagger 配置
# SWAGGER_URL = '/api/docs'
# API_URL = '/static/swagger.json'
# swagger_ui_blueprint = get_swaggerui_blueprint(
#     SWAGGER_URL,
#     API_URL,
#     config={
#         'app_name': "Blockchain Research Assistant API"
#     }
# )

# 初始化问答系统
qa_system = BlockchainQA()

paper_service = PaperService()
qa_service = QAService()

@bp.route('/')
def index():
    """API 健康检查"""
    return jsonify({
        "message": "API is running",
        "status": "healthy"
    })

@bp.route('/qa', methods=['POST'])
def answer_question():
    """回答问题"""
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({
                'error': 'No question provided'
            }), 400
            
        answer = qa_system.get_answer(question)
        return jsonify({
            'question': question,
            'answer': answer
        })
        
    except Exception as e:
        logger.error(f"Error in QA endpoint: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@bp.route('/analyze', methods=['POST'])
def analyze_text():
    """分析文本"""
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({
                'error': 'No text provided'
            }), 400
            
        # TODO: 实现文本分析功能
        return jsonify({
            'message': 'Text analysis endpoint'
        })
        
    except Exception as e:
        logger.error(f"Error in analysis endpoint: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@bp.route('/health')
def health_check():
    """健康检查接口"""
    try:
        return jsonify({
            'status': 'healthy',
            'message': 'API is running'
        })
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@bp.route('/test')
def test():
    """测试接口"""
    return jsonify({
        "message": "Test endpoint is working"
    }), 200

@bp.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        # 确保文件名安全
        filename = secure_filename(file.filename)
        
        # 确保上传目录存在
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        
        # 保存文件
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        # 验证文件是否成功保存
        if not os.path.exists(file_path):
            return jsonify({'error': 'Failed to save file'}), 500
            
        logger.info(f"File uploaded successfully: {filename}")
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename
        }), 200

    # 触发版本更新和增量训练
    from app.services.version_control import VersionControlSystem
    from app.tasks.training import incremental_train
    
    VersionControlSystem().commit_change('文献上传更新')
    threading.Thread(target=incremental_train).start()
    logger.info('文件处理完成，已触发模型更新')
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route("/summarize", methods=["POST"])
def summarize_text():
    """文本摘要接口"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or "text" not in data:
            logger.error("No text provided")
            return jsonify({"error": "No text provided"}), 400
        
        text = data["text"]
        max_length = data.get("max_length", 130)
        min_length = data.get("min_length", 30)
        
        # 生成摘要
        summary = summarizer.generate_summary(
            text,
            max_length=max_length,
            min_length=min_length
        )
        
        logger.info("Summary generated successfully")
        return jsonify({
            "summary": summary
        }), 200
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route("/search", methods=["POST"])
def search_documents():
    """文档搜索接口"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or "query" not in data:
            logger.error("No query provided")
            return jsonify({"error": "No query provided"}), 400
        
        query = data["query"]
        k = data.get("k", 5)
        
        # 生成查询向量
        query_vector = embedding_generator.generate_embedding(query).cpu().numpy()
        
        # 搜索文档
        distances, indices = document_retriever.search(query_vector, k=k)
        
        # 如果没有结果，返回空数组
        if len(indices) == 0:
            return jsonify({
                "distances": [],
                "indices": []
            }), 200
        
        logger.info("Document search completed successfully")
        return jsonify({
            "distances": distances.tolist(),
            "indices": indices.tolist()
        }), 200
        
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route("/embed", methods=["POST"])
def generate_embeddings():
    """生成文本嵌入接口"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or "texts" not in data:
            logger.error("No texts provided")
            return jsonify({"error": "No texts provided"}), 400
        
        texts = data["texts"]
        if not isinstance(texts, list):
            texts = [texts]
        
        # 生成嵌入
        embeddings = embedding_generator.generate_embeddings(texts)
        
        logger.info("Embeddings generated successfully")
        return jsonify({
            "embeddings": embeddings.tolist()
        }), 200
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/test_summary', methods=['POST'])
def test_summary():
    """测试摘要功能"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        from app.nlp import get_summarizer
        summarizer = get_summarizer()
        summary = summarizer.generate_summary(text)
        
        return jsonify({
            "original": text,
            "summary": summary
        })
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/analysis/<file_id>', methods=['GET'])
def get_analysis_status(file_id):
    """获取分析状态"""
    try:
        # 确保返回正确的 Content-Type
        status = {
            'status': 'processing',  # Default status
            'progress': 0,
            'current_task': 'Initializing analysis...'
        }
        
        # 从缓存中获取实际状态
        from app.utils.analysis import get_analysis_status
        cached_status = get_analysis_status(file_id)
        
        if cached_status:
            status.update(cached_status)
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"获取分析状态时出错: {e}")
        return jsonify({
            'status': 'error',
            'error': f'获取分析状态时出错: {str(e)}'
        }), 500

@bp.route('/blockchain/qa', methods=['POST'])
def blockchain_qa():
    """区块链问答接口"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
            
        question = data['question']
        if not question.strip():
            return jsonify({'error': 'Question cannot be empty'}), 400
            
        qa_system = BlockchainQA()
        answer = qa_system.get_answer(question)
        
        if not answer:
            return jsonify({'error': 'No relevant answer found'}), 404
            
        return jsonify({'answer': answer})
        
    except Exception as e:
        logger.error(f"QA error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# 添加 CORS 支持
@bp.after_request
def after_request(response):
    """处理 CORS 和 OPTIONS 请求"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 添加 OPTIONS 请求处理
@bp.route('/analysis/<file_id>', methods=['OPTIONS'])
@bp.route('/upload', methods=['OPTIONS'])
def handle_options(file_id=None):
    """处理 OPTIONS 请求"""
    return jsonify({}), 200

# 错误处理
@bp.errorhandler(413)
def request_entity_too_large(error):
    """
    处理请求实体过大的错误。
    """
    logger.error(f"Request entity too large: {error}")
    return jsonify({
        "error": "File too large",
        "max_size": MAX_CONTENT_LENGTH
    }), 413

@bp.route('/papers/<int:paper_id>/analysis', methods=['GET'])
def get_analysis(paper_id):
    """Get paper analysis results"""
    try:
        paper = paper_service.get_paper(paper_id)
        return jsonify(paper.to_dict())
    except Exception as e:
        logger.error(f"Failed to get analysis: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/papers/<int:paper_id>/refresh', methods=['POST'])
def refresh_analysis(paper_id):
    """Refresh paper analysis"""
    try:
        paper = paper_service.get_paper(paper_id)
        results = paper_service.analyze_paper(paper.id, paper.file_path)
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        logger.error(f"Failed to refresh analysis: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/qa/ask', methods=['POST'])
def ask_question():
    """Handle Q&A requests"""
    try:
        data = request.get_json()
        question = data.get('question')
        paper_id = data.get('paper_id')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        answer = qa_service.get_answer(question, paper_id)
        return jsonify({
            'success': True,
            'answer': answer
        })
        
    except Exception as e:
        logger.error(f"Failed to process question: {e}")
        return jsonify({'error': str(e)}), 500

# 添加文件分析接口
@bp.route('/analyze/<file_id>', methods=['POST'])
def analyze_file(file_id):
    """分析上传的文件"""
    try:
        # 获取文件路径
        file_path = os.path.join(UPLOAD_FOLDER, file_id)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        # 开始分析
        analysis_result = paper_service.analyze_paper(file_id, file_path)
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

# 添加趋势分析接口
@bp.route('/trend', methods=['GET'])
def get_trend():
    """获取研究趋势数据"""
    try:
        trend_data = paper_service.get_trend_data()
        return jsonify(trend_data)
    except Exception as e:
        logger.error(f"Failed to get trend data: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/kb/status', methods=['GET'])
def get_kb_status():
    """获取知识库状态"""
    try:
        qa_system = BlockchainQA()
        return jsonify({
            'status': 'success',
            'data': {
                'document_count': len(qa_system.knowledge_base),
                'paragraph_count': len(qa_system.knowledge_base) if qa_system.knowledge_base else 0,
                'last_updated': os.path.getmtime(os.path.join(qa_system.kb_dir, 'knowledge_base.json')) 
                if os.path.exists(os.path.join(qa_system.kb_dir, 'knowledge_base.json')) else None
            }
        })
    except Exception as e:
        logger.error(f"获取知识库状态失败: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@bp.route('/papers/<int:paper_id>/status')
def check_analysis_status(paper_id):
    """检查论文分析状态"""
    try:
        paper = Paper.query.get_or_404(paper_id)
        return jsonify({
            'status': paper.analysis_status,
            'progress': 100 if paper.analysis_status == 'completed' else 50,
            'error': paper.error_message,
            'current_task': 'Analysis completed' if paper.analysis_status == 'completed' 
                          else 'Analyzing' if paper.analysis_status == 'processing'
                          else 'Pending analysis'
        })
    except Exception as e:
        logger.error(f"Error checking paper status: {e}")
        return jsonify({'error': str(e)}), 500

# 导出 swagger_ui_blueprint 供 app/__init__.py 使用
__all__ = ['bp']

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == 'pdf'