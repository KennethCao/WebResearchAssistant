import matplotlib.pyplot as plt
try:
    import networkx as nx
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])
    import networkx as nx

from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def plot_line_chart(data, title="Line Chart", xlabel="X-axis", ylabel="Y-axis", save_path=None):
    """
    Draw line chart
    :param data: Dictionary with x values as keys and y values as values
    :param title: Chart title
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param save_path: Path to save the chart (optional)
    """
    if not data:
        logger.error("Empty data")
        raise ValueError("Data cannot be empty")

    if not isinstance(data, dict):
        logger.error("Data must be a dictionary type")
        raise TypeError("Data must be a dictionary")

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(list(data.keys()), list(data.values()), label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"折线图已保存到: {save_path}")
        else:
            plt.show()
            
        plt.close()
    except Exception as e:
        logger.error(f"Failed to draw line chart: {e}")
        raise RuntimeError(f"Line chart creation failed: {str(e)}")

def plot_network_graph(graph, title="Network Graph", save_path=None):
    """
    Draw network graph
    :param graph: NetworkX graph object
    :param title: Chart title
    :param save_path: Path to save the chart (optional)
    """
    if not graph:
        logger.error("Empty graph object")
        raise ValueError("Graph object cannot be empty")

    if not isinstance(graph, nx.Graph):
        logger.error("Graph object must be a NetworkX graph")
        raise TypeError("Invalid graph object type")

    try:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', 
                node_size=1500, font_size=10, font_weight='bold')
        plt.title(title)
        
        if save_path:
            # 确保保存目录存在
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"网络图已保存到: {save_path}")
        else:
            plt.show()
            
        plt.close()
    except Exception as e:
        logger.error(f"Failed to draw network graph: {e}")
        raise RuntimeError(f"Network graph creation failed: {str(e)}")

# 示例用法
if __name__ == "__main__":
    try:
        # 示例折线图数据
        line_data = {
            1: 10,
            2: 15,
            3: 7,
            4: 12,
            5: 9
        }

        # 绘制折线图
        plot_line_chart(line_data, title="示例折线图", xlabel="X 轴", ylabel="Y 轴", save_path="line_chart.png")

        # 示例网络图数据
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

        # 绘制网络图
        plot_network_graph(G, title="示例网络图", save_path="network_graph.png")
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        raise