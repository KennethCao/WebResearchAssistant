import os
import sys

def patch_sentence_transformers():
    try:
        import sentence_transformers
        st_path = os.path.dirname(sentence_transformers.__file__)
        st_file = os.path.join(st_path, 'SentenceTransformer.py')
        
        with open(st_file, 'r') as f:
            content = f.read()
        
        # 替换导入语句
        new_import = '''try:
    from huggingface_hub import HfApi, HfFolder, Repository, hf_hub_url
    try:
        from huggingface_hub import cached_download
    except ImportError:
        from huggingface_hub.file_download import cached_download
except ImportError:
    def cached_download(*args, **kwargs):
        return None
    HfApi = HfFolder = Repository = hf_hub_url = None
'''
        content = content.replace(
            'from huggingface_hub import HfApi, HfFolder, Repository, hf_hub_url, cached_download',
            new_import
        )
        
        with open(st_file, 'w') as f:
            f.write(content)
            
        print("Successfully patched sentence_transformers")
    except Exception as e:
        print(f"Failed to patch sentence_transformers: {e}")

if __name__ == '__main__':
    patch_sentence_transformers() 