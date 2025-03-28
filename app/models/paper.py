from datetime import datetime
from app.extensions import db

class Paper(db.Model):
    """论文模型"""
    __tablename__ = 'papers'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500))
    title = db.Column(db.String(500))
    summary = db.Column(db.Text)
    keywords = db.Column(db.Text)
    topics = db.Column(db.Text)
    content = db.Column(db.Text)
    analysis_status = db.Column(db.String(20), default='pending')
    error_message = db.Column(db.Text)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    added_to_kb = db.Column(db.Boolean, default=False)
    
    def __init__(self, **kwargs):
        super(Paper, self).__init__(**kwargs)
        if not self.uploaded_at:
            self.uploaded_at = datetime.utcnow()
    
    def __repr__(self):
        return f'<Paper {self.filename}>'
        
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'title': self.title,
            'summary': self.summary,
            'keywords': self.keywords,
            'topics': self.topics,
            'analysis_status': self.analysis_status,
            'uploaded_at': self.uploaded_at.strftime('%Y-%m-%d %H:%M:%S') if self.uploaded_at else None,
            'added_to_kb': bool(self.added_to_kb)
        } 