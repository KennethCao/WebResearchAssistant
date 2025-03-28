Blockchain Research Assistant
A comprehensive research assistant for analyzing, summarizing, and managing blockchain-related academic papers. This project utilizes state-of-the-art NLP technologies with local model inference support for offline environments.

Features
File Processing:
- Upload PDF academic papers in blockchain domain
- Extract metadata (title, authors, abstract)

NLP Capabilities:
- Automatic paper summarization
- Keyphrase extraction
- Similarity detection between papers
- Topic modeling and trend analysis

Retrieval Augmented Generation (RAG):
- FAISS-based vector storage and retrieval
- Semantic search over uploaded papers

Clustering & Network Analysis:
- KMeans-based topic clustering
- Interactive topic network visualization

Paper Evaluation:
- Relevance and quality scoring
- Intelligent improvement suggestions

Offline Support:
- Local pretrained model loading
- Efficient local semantic search with FAISS
30: Tech Stack
31: Backend
33: Flask (providing API services)
34: Frontend
36: Bootstrap (for styling and responsive design)
37: Database
38: SQLite (for storing metadata and analysis results)
39: Natural Language Processing
40: Hugging Face Transformers (e.g. Sentence-Transformer)
41: FAISS (for vector search)
43: Visualization
45: NetworkX (for topic network analysis)
46: Project Structure
48: [Code Structure]
50: ├── .gitignore                      # Git ignore file
51: ├── LICENSE                         # Project license
52: ├── README.md                       # Project documentation
53: ├── requirements.txt                # Python dependencies
54: ├── docker-compose.yml              # Docker configuration (optional)
55: ├── main.py                         # Main entry point
57: ├── app/                            # Core application code
58: │   ├── config/                     # Configuration files
59: │   ├── database/                   # Database models and migration scripts
60: │   ├── file_processing/            # File upload & PDF processing
61: │   ├── model_management/           # Model management (loading, downloading, fine-tuning)
62: │   ├── nlp/                        # NLP module (summarization, embeddings)
63: │   ├── clustering/                 # Clustering & topic analysis
64: │   ├── rag/                        # Retrieval Augmented Generation
65: │   ├── services/                   # Service layer (business logic)
66: │   ├── evaluation/                 # Paper scoring & improvement suggestions
67: │   ├── utils/                      # Utilities (logging, visualization)
68: │   ├── logs/                       # Log files
69: │   ├── version_control/            # Version control module
70: │   └── training_services/          # Training services module
72: ├── static/                         # Static files
73: │   ├── pdfs/                       # Uploaded PDFs
74: │   ├── outputs/                    # Analysis results
75: │   └── assets/                     # Static assets
77: ├── models/                         # Local models
78: │   ├── all-MiniLM-L6-v2/           # Sentence-Transformer model
79: │   ├── fine_tuned_model/           # Fine-tuned models
80: │   └── cache/                      # Model cache
82: ├── templates/                      # HTML templates
84: ├── docs/                           # Documentation
85: │   ├── architecture.md             # System architecture
86: │   ├── deployment.md               # Deployment guide
87: │   ├── api_docs.md                 # API documentation
88: │   ├── user_guide.md               # User manual
89: │   └── testing.md                  # Testing documentation
90: ├── tests/                          # Tests
91: │   ├── unit_tests/                 # Unit tests
92: │   └── integration_tests/          # Integration tests
93: └── scripts/                        # Scripts
94: ├── automation/                 # Automation scripts
95: └── monitoring/                # Monitoring scripts
96: Installation Steps
97: 1. Clone repository
103: 2. Environment Setup
104: Run auto-configuration script:
106: # Windows
109: # Other systems
112: The script will automatically:
113: - Create Python virtual environment
114: - Install requirements.txt dependencies
115: - Set environment variables (FLASK_APP=main.py, FLASK_ENV=development, PORT=5000)
117: 3. Database Initialization
122: 4. First Run Process
123: The system will automatically:
128: 1. Check models directory for base models (all-MiniLM-L6-v2)
129: 2. Download missing models to models/
130: 3. Trigger incremental training with new data
131: 4. Maintain manual training entry points:
134: python scripts/train_model.py --full  # Full training
135: python scripts/train_model.py --incremental  # Incremental training
138: 5. Access Application
139: After successful startup, access via:
142: Note:
143: - First run requires downloading base models (~400MB)
144: - Incremental training runs daily at midnight
145: - Manual training recommended on GPU
146: Access via http://127.0.0.1:5000
148: User Guide
149: File Upload
150: Access http://127.0.0.1:5000
151: Upload blockchain-related PDFs
152: Analysis Features
153: View extracted metadata (title, author, abstract)
154: Auto-generated summaries
155: Keyword and trend extraction
156: Topic clustering & similarity analysis
157: Offline Mode
158: Requirements:
160: - All required models in models/
161: - FAISS index in static/outputs/
162: Contributing
163: Fork repository
164: Create branch: git checkout -b feature-branch-name
165: Commit changes: git commit -m "Add new feature"
166: Push branch: git push origin feature-branch-name
167: Submit Pull Request
1. Create virtual environment
```bash
python -m venv venv
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Initialize database
```bash
python scripts/setup_db.py
```

4. Start application
```bash
python main.py
```
