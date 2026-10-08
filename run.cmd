@echo off
setlocal
cd /d "%~dp0"
if not exist "build-debug\Debug\vulkan-starter-app.exe" (
    echo Build the project with build.cmd first.
    pause
    exit /b 1
)
"build-debug\Debug\vulkan-starter-app.exe"
if errorlevel 1 pause
