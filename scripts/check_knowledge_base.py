from app.nlp.blockchain_qa import BlockchainQA
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def check_knowledge_base():
    """检查知识库状态"""
    try:
        qa_system = BlockchainQA()
        
        # 打印知识库信息
        print(f"知识库中的段落数量: {len(qa_system.knowledge_base)}")
        
        # 打印一些示例段落
        print("\n示例段落:")
        for i, para in enumerate(qa_system.knowledge_base[:3]):
            print(f"\n段落 {i+1}:")
            print(para[:200] + "...")
            
        # 测试问答功能
        test_questions = [
            "什么是区块链?",
            "区块链有哪些应用场景?",
            "智能合约是什么?"
        ]
        
        print("\n测试问答:")
        for question in test_questions:
            print(f"\nQ: {question}")
            answer = qa_system.get_answer(question)
            print(f"A: {answer}")
            
    except Exception as e:
        logger.error(f"检查知识库时出错: {str(e)}")

if __name__ == "__main__":
    check_knowledge_base() 