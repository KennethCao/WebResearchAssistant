import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.utils.logger import setup_logger
from app.utils.file_utils import create_directory

# 设置日志记录
logger = setup_logger()

def build_topic_network(data, labels, top_n=10):
    """
    构建主题网络图。
    :param data: 输入数据（嵌入向量或文本列表）
    :param labels: 聚类结果的标签
    :param top_n: 每个主题选择的关键节点数量
    :return: NetworkX 图对象
    """
    try:
        logger.info("开始构建主题网络图")

        # 创建 NetworkX 图
        graph = nx.Graph()

        # 按聚类标签分组数据
        cluster_data = defaultdict(list)
        for label, item in zip(labels, data):
            cluster_data[label].append(item)

        # 添加节点和边
        for cluster, items in cluster_data.items():
            # 限制每个主题的节点数为 top_n
            limited_items = items[:top_n]

            # 计算节点对之间的相似度
            if isinstance(data[0], np.ndarray):  # 如果是嵌入向量
                similarity_matrix = cosine_similarity(limited_items)
            else:  # 如果是文本数据，默认占位相似度
                similarity_matrix = np.full((len(limited_items), len(limited_items)), 0.5)

            for i, item in enumerate(limited_items):
                node_id = f"Cluster {cluster} - Node {i}"
                graph.add_node(node_id, cluster=cluster, content=item)

                for j in range(i + 1, len(limited_items)):
                    similarity = similarity_matrix[i, j]
                    other_node_id = f"Cluster {cluster} - Node {j}"
                    graph.add_edge(node_id, other_node_id, weight=similarity)

        logger.info("主题网络图构建完成")
        return graph
    except Exception as e:
        logger.error(f"构建主题网络图失败: {e}")
        raise RuntimeError(f"构建主题网络图失败: {str(e)}")

def visualize_topic_network(graph, save_path=None):
    """
    可视化主题网络图。
    :param graph: NetworkX 图对象
    :param save_path: 保存路径（如果为空，则直接显示图形）
    """
    try:
        logger.info("开始绘制主题网络图")

        plt.figure(figsize=(12, 8))

        # 获取节点颜色
        clusters = nx.get_node_attributes(graph, 'cluster')
        color_map = [clusters[node] for node in graph.nodes]

        # 计算节点位置
        pos = nx.spring_layout(graph, k=0.3, iterations=50)

        # 绘制节点
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=color_map,
            cmap=plt.cm.tab20,
            node_size=800,
            alpha=0.8
        )

        # 绘制边
        nx.draw_networkx_edges(
            graph,
            pos,
            width=1.0,
            alpha=0.5
        )

        # 绘制节点标签
        nx.draw_networkx_labels(
            graph,
            pos,
            font_size=10,
            font_family="sans-serif"
        )

        # 绘制边标签（相似度）
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_color='red'
        )

        # 设置标题
        plt.title("主题网络图", fontsize=16)

        # 保存或显示
        if save_path:
            create_directory(os.path.dirname(save_path))
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"主题网络图已保存到: {save_path}")
        else:
            plt.show()

        logger.info("主题网络图绘制完成")
    except Exception as e:
        logger.error(f"绘制主题网络图失败: {e}")
        raise RuntimeError(f"绘制主题网络图失败: {str(e)}")