@echo off
REM 区块链研究助手数据库初始化脚本
REM 版本: 2.0 | 最后更新: 2024-03-15

setlocal enabledelayedexpansion

REM ─── 初始化日志配置 ───
set LOG_FILE=logs\db_init.log
if not exist logs mkdir logs

REM ─── 创建数据库目录 ───
if not exist app\database mkdir app\database

REM ─── 初始化数据库 ───
echo [数据库] 正在初始化SQLite数据库...
python -c "from app.database import init_db; init_db()" >> %LOG_FILE% 2>&1
if errorlevel 1 (
  echo [错误] 数据库初始化失败 >> %LOG_FILE%
  exit /b 1
)

REM ─── 验证数据库文件 ───
if not exist app\database\research_papers.db (
  echo [错误] 数据库文件未创建 >> %LOG_FILE%
  exit /b 1
)

echo [成功] 数据库初始化完成
endlocal

exit /b 0