import os
import requests
import shutil

def download_swagger_ui():
    """下载 Swagger UI 静态资源文件"""
    base_url = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0"
    files = [
        'swagger-ui.css',
        'swagger-ui-bundle.js',
        'swagger-ui-standalone-preset.js',
        'favicon-32x32.png'
    ]
    
    # 确保目标目录存在
    static_dir = os.path.join('app', 'static', 'dist')
    os.makedirs(static_dir, exist_ok=True)
    
    for file in files:
        url = f"{base_url}/{file}"
        target_path = os.path.join(static_dir, file)
        
        print(f"Downloading {file}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print(f"Successfully downloaded {file}")
        else:
            print(f"Failed to download {file}")

if __name__ == "__main__":
    download_swagger_ui() 