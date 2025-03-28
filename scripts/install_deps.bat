@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul

REM Blockchain Research Assistant Dependency Installer
REM Version: 2.0 | Last Updated: 2024-03-15

REM === Initialize Log Configuration ===
set LOG_FILE=logs\install.log
if not exist logs mkdir logs

REM === Configure PyPI Mirror ===
echo [Dependency] Configuring Tsinghua PyPI mirror...
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >> %LOG_FILE% 2>&1
if errorlevel 1 (
  echo [ERROR] Mirror configuration failed >> %LOG_FILE%
  exit /b 1
)

REM === Install Base Dependencies ===
echo [Dependency] Installing Python dependencies...
pip install -r requirements.txt >> %LOG_FILE% 2>&1
if errorlevel 1 (
  echo [ERROR] Dependency installation failed >> %LOG_FILE%
  exit /b 1
)

REM === Check Dependency Conflicts ===
echo [Dependency] 正在检查依赖冲突...
pip check >> %LOG_FILE% 2>&1
if errorlevel 1 (
  echo [ERROR] 发现依赖冲突，请检查requirements.txt >> %LOG_FILE%
  exit /b 1
)

REM === Verify Installation ===
echo [Dependency] Verifying installation...
pip freeze | findstr /C:"Flask==" /C:"transformers==" >> %LOG_FILE%
if errorlevel 1 (
  echo [WARNING] Some dependencies not installed properly >> %LOG_FILE%
)

echo [SUCCESS] Dependencies installed successfully
endlocal

exit /b 0