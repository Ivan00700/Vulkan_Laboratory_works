@echo off
setlocal
cd /d "%~dp0"
if not defined VULKAN_SDK set "VULKAN_SDK=C:\VulkanSDK\1.4.363.0"
if not exist "%VULKAN_SDK%\Bin\glslc.exe" (
    echo Vulkan SDK was not found. Check the VULKAN_SDK environment variable.
    pause
    exit /b 1
)
set "PATH=%VULKAN_SDK%\Bin;%PATH%"
cmake --preset msvc-debug -A x64
if errorlevel 1 goto failed
cmake --build build-debug --config Debug --parallel 4
if errorlevel 1 goto failed
echo Build succeeded. Run run.cmd to start the application.
exit /b 0
:failed
echo Build failed. Read the error above.
pause
exit /b 1
