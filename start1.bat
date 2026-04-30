@echo off
setlocal

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8001"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo [1/2] Starting backend (FastAPI, port 8001)...
cd /d "%~dp0backend"
start "Backend" cmd /k "python -m uvicorn main:app --port 8001 --reload"
timeout /t 3 /nobreak >nul

echo [2/2] Starting frontend (Vite, port 5173)...
cd /d "%~dp0frontend"
start "Frontend" cmd /k "set PATH=C:\Program Files\nodejs;%APPDATA%\npm;%PATH% & npm run dev"

echo.
echo Backend : http://localhost:8001
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8001/docs
echo.
timeout /t 3 /nobreak >nul
start http://localhost:5173

endlocal