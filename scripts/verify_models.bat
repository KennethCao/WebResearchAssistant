@echo off
REM 区块链研究助手模型验证脚本
REM 版本: 2.0 | 最后更新: 2024-03-15

setlocal enabledelayedexpansion

REM ─── 初始化日志配置 ───
set LOG_FILE=logs\models.log
if not exist logs mkdir logs

REM ─── 创建模型目录 ───
if not exist models mkdir models

REM ─── 验证基础模型 ───
echo [模型检查] 正在验证基础模型完整性...
if not exist models\all-MiniLM-L6-v2\pytorch_model.bin (
  echo [下载] 正在下载基础模型...
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" >> %LOG_FILE% 2>&1
  if errorlevel 1 (
    echo [错误] 模型下载失败 >> %LOG_FILE%
    exit /b 1
  )
)

REM ─── 验证FAISS索引 ───
echo [模型检查] 正在验证FAISS索引...
if not exist static\outputs\faiss_index\index.faiss (
  echo [初始化] 创建新的FAISS索引...
  python -c "from app.rag import init_faiss; init_faiss()" >> %LOG_FILE% 2>&1
  if errorlevel 1 (
    echo [错误] FAISS初始化失败 >> %LOG_FILE%
    exit /b 1
  )
)

echo [成功] 模型验证完成
endlocal

exit /b 0