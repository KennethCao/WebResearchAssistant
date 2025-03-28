import os

def list_files_in_directory(directory):
    """
    列出指定目录下的所有文件路径并读取文件内容
    :param directory: 目录路径
    :return: 文件内容列表
    """
    file_contents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"#file:{file_path}")  # 打印文件完整路径
    return file_contents

# 示例调用
if __name__ == "__main__":
    directory_path = "D:\\blockchain-research-assistant"  # 修改为实际的目录路径
    print(f"开始读取目录: {directory_path}")  # 添加日志
    contents = list_files_in_directory(directory_path)
    for content in contents:
        print(content)  # 打印文件内容
    print(f"完成读取目录: {directory_path}")  # 添加日志
