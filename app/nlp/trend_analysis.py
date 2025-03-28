from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from app.utils.logger import setup_logger
from app.nlp.keyword_extractor import extract_keywords

# 设置日志记录器，提供模块名称
logger = setup_logger(__name__)

def analyze_trend(texts: List[str], dates: List[str] = None) -> Dict[str, Any]:
    """
    分析文本集合中的趋势
    
    Args:
        texts: 文本列表
        dates: 对应的日期列表（可选）
        
    Returns:
        包含趋势分析结果的字典
    """
    try:
        # 如果没有提供日期，使用索引作为时间序列
        if dates is None:
            dates = [str(i) for i in range(len(texts))]
            
        # 提取所有文本的关键词
        all_keywords = []
        for text in texts:
            keywords = extract_keywords(text, num_keywords=5, method='hybrid')
            all_keywords.extend(keywords)
            
        # 计算关键词频率
        keyword_freq = Counter(all_keywords)
        
        # 创建时间序列数据
        trend_data = []
        for i, text in enumerate(texts):
            keywords = extract_keywords(text, num_keywords=5, method='hybrid')
            trend_data.append({
                'date': dates[i],
                'keywords': keywords,
                'count': len(keywords)
            })
            
        # 转换为 DataFrame 以便进行时间序列分析
        df = pd.DataFrame(trend_data)
        
        # 计算趋势指标
        result = {
            'top_keywords': dict(keyword_freq.most_common(10)),
            'keyword_count_over_time': df['count'].tolist(),
            'dates': dates,
            'total_keywords': len(all_keywords),
            'unique_keywords': len(set(all_keywords))
        }
        
        logger.info("Trend analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        return {
            'error': str(e),
            'top_keywords': {},
            'keyword_count_over_time': [],
            'dates': [],
            'total_keywords': 0,
            'unique_keywords': 0
        }

def plot_trend(trend_df, title="关键词趋势分析", save_path=None):
    """
    绘制关键词趋势图。
    :param trend_df: 包含关键词频率趋势的 Pandas DataFrame
    :param title: 图表标题
    :param save_path: 图表保存路径（如果为 None，则直接显示）
    """
    if trend_df.empty:
        logger.error("趋势数据为空")
        raise ValueError("趋势数据为空")

    try:
        logger.info("开始绘制趋势图")
        plt.figure(figsize=(10, 6))
        for keyword in trend_df.columns:
            plt.plot(trend_df.index, trend_df[keyword], label=keyword)

        plt.title(title)
        plt.xlabel("时间")
        plt.ylabel("关键词频率")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            logger.info(f"趋势图已保存到: {save_path}")
        else:
            plt.show()

        logger.info("趋势图绘制完成")
    except Exception as e:
        logger.error(f"绘制趋势图失败: {e}")
        raise RuntimeError(f"绘制趋势图失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 测试代码
    test_texts = [
        "Blockchain technology revolutionizes digital transactions",
        "Smart contracts enable automated execution",
        "Cryptocurrencies provide new financial paradigms"
    ]
    test_dates = ["2023-01", "2023-02", "2023-03"]
    
    result = analyze_trend(test_texts, test_dates)
    print("Trend Analysis Result:")
    print(f"Top Keywords: {result['top_keywords']}")
    print(f"Keyword Count Over Time: {result['keyword_count_over_time']}")

    # 绘制趋势图
    plot_trend(trend_df, title="区块链关键词趋势分析", save_path="trend_analysis.png")