@echo off
REM 区块链研究助手环境变量设置脚本
REM 版本: 2.0 | 最后更新: 2024-03-15

setlocal enabledelayedexpansion

REM ─── 初始化日志配置 ───
set LOG_FILE=logs\env_setup.log
if not exist logs mkdir logs

REM ─── 标准化环境变量 ───
echo [环境变量] 正在设置系统环境变量...
(
  echo FLASK_APP=main.py
  echo FLASK_ENV=development
  echo FAISS_INDEX_PATH=static/outputs/faiss_index
  echo MODEL_DIR=models
  echo DB_PATH=app/database/research_papers.db
) > .env

REM ─── 验证环境变量 ───
if not exist .env (
  echo [错误] 环境变量文件创建失败 >> %LOG_FILE%
  exit /b 1
)

echo [成功] 环境变量配置完成
endlocal

exit /b 0