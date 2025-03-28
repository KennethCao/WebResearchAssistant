@echo off

rem 强制删除虚拟环境
echo Force deleting virtual environment...
rmdir /s /q "%CD%\venv" 2>nul
if exist "%CD%\venv" (
    echo [ERROR] Failed to delete virtual environment
    exit /b 1
)

rem 创建新虚拟环境
echo Creating new virtual environment...
set PYTHON=C:\Users\an\AppData\Local\Programs\Python\Python39\python.exe
%PYTHON% -m venv "%CD%\venv" || exit /b 1
call "%CD%\venv\Scripts\activate.bat" || exit /b 1

rem 设置FLASK密钥环境变量
set ROOT_DIR=%~sdp0
set "DB_DIR=%ROOT_DIR%\instance"
set "LOG_DIR=%ROOT_DIR%\logs"

python -c "import os, sys; from pathlib import Path; root_path = Path(sys.argv[1]).resolve().as_posix().replace('/', '\\'); os.environ.update({'ROOT_DIR': root_path, 'SECRET_KEY': os.environ.get('SECRET_KEY')})" "%ROOT_DIR:~0,-1%"
taskkill /F /IM python* /T >nul 2>&1
taskkill /F /IM pythonw* /T >nul 2>&1
taskkill /F /IM pip* /T >nul 2>&1
for /l %%i in (1,1,5) do (
  wmic process where "name like 'python%'" delete >nul 2>&1
  wmic process where "name like 'pythonw%'" delete >nul 2>&1
  timeout /t 45 /nobreak >nul
)

rem Force install dependencies
echo Installing dependencies...
taskkill /F /IM python* /T >nul 2>&1
taskkill /F /IM pip* /T >nul 2>&1
wmic process where "name like 'python%'" delete >nul 2>&1
wmic process where "name like 'pip%'" delete >nul 2>&1
timeout /t 45 /nobreak >nul
echo Checking process status...
tasklist | findstr /i "python pip" && (echo [ERROR] 仍有残留进程未终止 && goto CLEAN_PROCS)
:CLEAN_PROCS
wmic process where "name like 'python%'" delete >nul 2>&1
wmic process where "name like 'pip%'" delete >nul 2>&1
timeout /t 5 /nobreak >nul
tasklist | findstr /i "python pip" && (echo [ERROR] 二次清理后仍有残留进程 && exit /b 1)
attrib -r "%CD%\venv\Scripts\pip*.exe" >nul 2>&1
attrib -r "%CD%\venv\Scripts\pip*.exe" >nul 2>&1
timeout /t 3 /nobreak >nul
attrib -r "%CD%\venv\Scripts\pip3.exe" >nul 2>&1
for /l %%i in (1,1,5) do (
  if exist "%CD%\venv\Scripts\pip3.exe" (
    taskkill /F /IM python* /T >nul 2>&1
    taskkill /F /IM pip* /T >nul 2>&1
    wmic process where "name like 'python%'" delete >nul 2>&1
    wmic process where "name like 'pip%'" delete >nul 2>&1
    timeout /t 45 /nobreak >nul
    attrib -r "%CD%\venv\Scripts\pip*.exe" >nul 2>&1
:CleanProcs
for /l %%i in (1,1,3) do (
  taskkill /F /IM python* /T >nul 2>&1
  taskkill /F /IM pythonw* /T >nul 2>&1
  wmic process where "name like 'python%'" delete >nul 2>&1

  rem 使用handle.exe检测文件句柄
  handle.exe -nobanner -accepteula "%CD%\venv\Scripts\pip3.exe" 2>nul | findstr /i "pid:" >handles.txt
  if %errorlevel% equ 0 (
    for /f "tokens=2" %%p in (handles.txt) do taskkill /F /PID %%p >nul 2>&1
    del handles.txt
    timeout /t 15 /nobreak >nul
  )
  timeout /t 10 /nobreak >nul
)

:CheckHandles
handle.exe -nobanner -accepteula "%CD%\venv\Scripts\pip3.exe" 2>nul | findstr /i "pid:" >handles.txt
if %errorlevel% equ 0 (
  for /f "tokens=2" %%p in (handles.txt) do taskkill /F /PID %%p >nul 2>&1
  del handles.txt
  timeout /t 15 /nobreak >nul
  goto CheckHandles
)

:ForceDelete
for /l %%i in (1,1,5) do (
  if exist "%CD%\venv\Scripts\pip3.exe" (
    attrib -r -s -h "%CD%\venv\Scripts\pip3.exe"
    del /f /q "%CD%\venv\Scripts\pip3.exe" && goto Deleted
    timeout /t 8 /nobreak >nul
  )
)
echo [DEBUG] 最终删除失败，当前句柄信息：
handle.exe -nobanner "%CD%\venv\Scripts\pip3.exe"
exit /b 1

:Deleted
attrib -r "%CD%\venv\Scripts\pip3.exe" >nul 2>&1
if exist "%CD%\venv\Scripts\pip3.exe" (
    del /f /q "%CD%\venv\Scripts\pip3.exe" >nul 2>&1 || (
        echo [DEBUG] 文件删除失败，当前句柄信息：
        handle.exe "%CD%\venv\Scripts\pip3.exe"
        exit /b 1
    )
)
attrib -r "%CD%\venv\Scripts\pip3.exe" >nul 2>&1
for /l %%i in (1,1,5) do (
  if exist "%CD%\venv\Scripts\pip3.exe" (
    taskkill /F /IM python* /T >nul 2>&1
    taskkill /F /IM pip* /T >nul 2>&1
    wmic process where "name like 'python%'" delete >nul 2>&1
    wmic process where "name like 'pip%'" delete >nul 2>&1
    timeout /t 45 /nobreak >nul
    attrib -r "%CD%\venv\Scripts\pip*.exe" >nul 2>&1
del /f /q "%CD%\venv\Scripts\pip3.exe" >nul 2>&1
  )
)
if exist "%CD%\venv\Scripts\pip3.exe" (
    echo [ERROR] pip3.exe 文件仍被锁定
    tasklist | findstr /i "python pip"
    exit /b 1
)
timeout /t 3 /nobreak >nul
attrib -r "%CD%\venv\Scripts\pip3.exe" >nul 2>&1
for /l %%i in (1,1,5) do (
  if exist "%CD%\venv\Scripts\pip3.exe" (
    taskkill /F /IM python* /T >nul 2>&1
    taskkill /F /IM pip* /T >nul 2>&1
    wmic process where "name like 'python%'" delete >nul 2>&1
    wmic process where "name like 'pip%'" delete >nul 2>&1
    timeout /t 45 /nobreak >nul
    attrib -r "%CD%\venv\Scripts\pip*.exe" >nul 2>&1
del /f /q "%CD%\venv\Scripts\pip3.exe" >nul 2>&1
  )
)
if exist "%CD%\venv\Scripts\pip3.exe" (
    echo [ERROR] pip3.exe 文件仍被锁定
    tasklist | findstr /i "python pip"
    exit /b 1
)
if exist "%CD%\venv\Scripts\pip*.exe" (
    echo [ERROR] 关键文件仍被锁定，请手动关闭所有Python相关进程后重试
    exit /b 1
)
%PYTHON% -m pip install --upgrade pip --retries 5 --retry-delay 10 || exit /b 1
pip install -r requirements.txt --force-reinstall --no-cache-dir --retries 3 --retry-delay 5 || exit /b 1

rem 初始化数据库迁移
echo Initializing database migration...
if not exist "%DB_DIR%" mkdir "%DB_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
python manage.py db init || exit /b 1
python manage.py db migrate -m "强制重建环境迁移" || exit /b 1
flask db upgrade || exit /b 1

rem 验证依赖完整性
echo Verifying dependencies...
%PYTHON% -m pip check || (
    echo [ERROR] Dependency conflicts detected
    exit /b 1
)

rem 验证数据库
%PYTHON% -m flask verify-db || (
    echo [ERROR] Database verification failed
    exit /b 1
)

echo Environment rebuilt successfully. Please run start.bat to launch the service