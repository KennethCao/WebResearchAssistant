# app/__init__.py
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import sys
import numpy as np

# 导入自定义模块
from app.config import ConfigLoader, validate_config
from app.file_processing.pdf_parser import parse_pdf
from app.file_processing.file_validator import validate_file
from app.nlp.summarizer import generate_summary
from app.nlp.keyword_extractor import extract_keywords
from app.nlp.embedding_generator import generate_embeddings, get_embedding_generator
from app.utils.logger import setup_logger
from app.clustering.kmeans_clustering import perform_kmeans_clustering as perform_clustering
from app.database import get_db, init_db
from app.nlp import load_default_model, get_model
from app.rag.retriever import FAISSRetriever
from app.utils.error_handler import setup_error_handlers

# 初始化 Flask 应用
app = Flask(__name__)

# 初始化日志记录
logger = setup_logger()

# 初始化 FAISS 检索器
faiss_retriever = None

# 工具函数：检查文件扩展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx', 'pptx'}

# 首页路由（文件上传页面）
@app.route('/')
def index():
    return render_template('blockchain_upload.html')

# 文件上传路由
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 验证 PDF 文件
        file_extension = filename.rsplit('.', 1)[1].lower()
        parser_map = {
            'pdf': parse_pdf,
            'docx': DocxParser().parse,
            'pptx': PPTParser().parse
        }
        
        if not validate_file(file_path) or file_extension not in parser_map:
            logger.error(f"File validation failed: {filename}")
            return jsonify({'error': 'Invalid file format or unsupported type'}), 400
        
        # 调用对应格式的解析器
        try:
            parsed_content = parser_map[file_extension](file_path)
            db.session.add(Document(
                filename=filename,
                content=parsed_content['text'],
                metadata=parsed_content['metadata']
            ))
            db.session.commit()

        logger.info(f"File uploaded successfully: {filename}")
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200

    return jsonify({'error': 'File type not supported'}), 400

# 提取元信息路由
@app.route('/extract_metadata', methods=['POST'])
def extract_metadata():
    file_path = request.json.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 400

    text, metadata = parse_pdf(file_path)
    logger.info(f"Metadata extraction successful: {metadata}")
    return jsonify({'metadata': metadata}), 200

# 生成摘要路由
@app.route('/generate_summary', methods=['POST'])
def generate_summary_route():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text content provided'}), 400

    summary = generate_summary(text)
    logger.info(f"Summary generation successful: {summary}")
    return jsonify({'summary': summary}), 200

# 提取关键词路由
@app.route('/extract_keywords', methods=['POST'])
def extract_keywords_route():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text content provided'}), 400

    keywords = extract_keywords(text)
    logger.info(f"Keyword extraction successful: {keywords}")
    return jsonify({'keywords': keywords}), 200

# 嵌入生成与检索路由
@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings_route():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text content provided'}), 400

    embeddings = generate_embeddings(text)
    logger.info(f"Embedding generation successful: {embeddings}")
    return jsonify({'embeddings': embeddings.tolist()}), 200

@app.route('/search_similar', methods=['POST'])
def search_similar_route():
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'No query content provided'}), 400

    # 使用 FAISS 检索器搜索
    distances, indices = faiss_retriever.search(query, k=5)

    # 检索结果
    results = [{'index': int(idx), 'distance': float(dist)} 
               for idx, dist in zip(indices[0], distances[0])]
    logger.info(f"Search results: {results}")
    return jsonify({'results': results}), 200

# 主题聚类与分析路由
@app.route('/cluster_analysis', methods=['POST'])
def cluster_analysis():
    texts = request.json.get('texts')
    if not texts or not isinstance(texts, list):
        return jsonify({'error': 'No valid text list provided'}), 400

    try:
        # 生成嵌入
        embeddings = [generate_embeddings(text) for text in texts]
        
        # 确保 embeddings 是 numpy 数组
        embeddings = np.array([e.cpu().numpy() for e in embeddings])
        
        # 执行聚类
        cluster_results = perform_clustering(embeddings)
        
        logger.info(f"Clustering analysis successful: {cluster_results}")
        return jsonify({'clusters': cluster_results}), 200
    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

# 错误处理路由
@app.errorhandler(404)
def page_not_found(e):
    logger.error(f"404 Error: {str(e)}")
    return jsonify({'error': 'Requested resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500 Error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

# 添加应用初始化检查
def initialize_app():
    try:
        # 1. Load and validate configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        app.config.update(config)
        if not validate_config(config):
            raise RuntimeError("Configuration validation failed")
        
        # 2. Initialize logging system
        logger.info("Starting application initialization...")

        # 3. Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['BLOCKCHAIN_PAPER_PATH'], exist_ok=True)
        os.makedirs('./static/outputs/blockchain_outputs', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./models/blockchain', exist_ok=True)
        os.makedirs('./static/outputs/summaries', exist_ok=True)
        os.makedirs('./static/outputs/keywords', exist_ok=True)
        os.makedirs('./static/outputs/embeddings', exist_ok=True)
        os.makedirs('./static/outputs/trend_analysis', exist_ok=True)
        os.makedirs('./static/outputs/network_graphs', exist_ok=True)
        
        # 4. Initialize database
        init_db()
        
        # 5. Initialize models and index
        load_default_model()
        
        # 初始化 embedding generator 和 FAISS 检索器
        global faiss_retriever
        device = 'cpu'  # 确保使用 CPU
        get_embedding_generator(device=device)
        faiss_retriever = FAISSRetriever(
            dimension=app.config['BLOCKCHAIN_EMBEDDING_DIMENSION'],
            index_path=app.config['BLOCKCHAIN_INDEX_PATH']
        )
        
        logger.info("Application initialization completed")
        setup_error_handlers(app)
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        sys.exit(1)

# 健康检查路由
@app.route('/health')
def health_check():
    """System health check"""
    try:
        # Check database connection
        db = get_db()
        next(db)
        
        # Check model status
        model = get_model()
        
        # Check FAISS index
        if faiss_retriever is None or faiss_retriever.index is None:
            raise RuntimeError("FAISS index not initialized")
            
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'model': 'loaded',
            'faiss': 'initialized'
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# 主函数
if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=app.config['DEFAULT_PORT'])