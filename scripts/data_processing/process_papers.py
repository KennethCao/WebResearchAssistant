process_papers.pyimport os
import pandas as pd
from app.file_processing.pdf_parser import parse_pdf
from app.config.environment import BASE_DIR
from app.nlp.blockchain_qa import BlockchainQA
import logging
from retry import retry
import threading

logger = logging.getLogger(__name__)

@retry(tries=3, delay=2, logger=logger)
def process_pdf_with_retry(file_path):
    return parse_pdf(file_path)

def process_papers():
    """Process blockchain papers and generate training data"""
    papers_data = []
    
    # Specify PDF directory
    pdf_dir = os.path.join(BASE_DIR, 'static', 'pdfs')
    
    # Ensure output directory exists
    data_dir = os.path.join(BASE_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Processing papers from: {pdf_dir}")
    
    # Initialize QA system
    qa_system = BlockchainQA()
    
    # Iterate through paper directory
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_dir, filename)
            try:
                print(f"Processing: {filename}")
                # Parse PDF
                text, metadata = process_pdf_with_retry(file_path)
                
                paragraphs = re.split(r'\n{2,}', text)
                valid_paras = 0
                for para in paragraphs:
                    if len(para.strip()) > 100:
                        # Blockchain keyword detection
                        if not any(keyword.lower() in para.lower() for keyword in ['blockchain', 'smart contract', 'decentralized']):
                            continue
                        # Add to training data
                        papers_data.append({
                            'text': para.strip(),
                            'label': 1,  # Blockchain-related text
                            'source': filename,
                            'title': metadata.get('title', ''),
                            'authors': metadata.get('authors', '')
                        })
                        
                        # Sync to knowledge base
                        if qa_system.is_blockchain_related(para.strip()):
                            qa_system.add_to_knowledge_base(para.strip(), source=filename)
                        valid_paras += 1
                
                print(f"Extracted {valid_paras} valid paragraphs from {filename}")
                        
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
    
    # Save training data
    df = pd.DataFrame(papers_data)
    output_path = os.path.join(data_dir, 'blockchain_papers.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info("Paper processing completed")
    
    # Save knowledge base updates
    qa_system.save_knowledge_base()
    logger.info(f"- Knowledge base updated, new paragraphs: {len(papers_data)}")
    
    # Trigger version update and incremental training
    from app.services.version_control import VersionControlSystem
    from app.tasks.training import incremental_train
    
    VersionControlSystem().commit_change('Add new paper processing')
    incremental_train()
    # Execute model training asynchronously
    # Asynchronous model training execution
    from app.commands.train import train_model
    # 异步执行模型训练
    train_thread = threading.Thread(target=train_model)
    train_thread.start()