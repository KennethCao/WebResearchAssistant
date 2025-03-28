import os
import sys

def fix_sentence_transformers():
    try:
        # 找到 sentence_transformers 包的位置
        import sentence_transformers
        st_path = os.path.dirname(sentence_transformers.__file__)
        st_file = os.path.join(st_path, 'SentenceTransformer.py')
        
        # 读取原文件内容
        with open(st_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 备份原文件
        backup_file = st_file + '.backup'
        if not os.path.exists(backup_file):
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 替换导入语句
        old_import = 'from huggingface_hub import HfApi, HfFolder, Repository, hf_hub_url, cached_download'
        new_import = '''try:
    from huggingface_hub import HfApi, HfFolder, Repository, hf_hub_url
    try:
        from huggingface_hub import cached_download
    except ImportError:
        try:
            from huggingface_hub.file_download import cached_download
        except ImportError:
            def cached_download(*args, **kwargs):
                return None
except ImportError:
    HfApi = HfFolder = Repository = hf_hub_url = None
    def cached_download(*args, **kwargs):
        return None'''
        
        # 替换内容
        new_content = content.replace(old_import, new_import)
        
        # 写入修改后的内容
        with open(st_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print("Successfully fixed sentence_transformers")
        return True
    except Exception as e:
        print(f"Failed to fix sentence_transformers: {e}")
        return False

if __name__ == '__main__':
    fix_sentence_transformers() 