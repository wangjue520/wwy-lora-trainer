@echo off
rem WWY LoRA 训练器 - 唯一入口。双击运行。
rem 本文件保存为 GBK 编码（cmd 默认按系统编码读 bat，存成 UTF-8 中文会乱码）。
setlocal
cd /d "%~dp0"
title WWY LoRA 训练器

rem 界面后台只用 Python 标准库，不需要装任何包；训练环境（PyTorch 等）在界面里一键安装
set "PY=%~dp0python\python.exe"
if exist "%PY%" goto run

echo.
echo   首次运行：下载免安装的便携版 Python 3.13（约 30 MB）到
echo       %~dp0python
echo   不会改动系统里已有的 Python。
echo.
where powershell >nul 2>nul
if errorlevel 1 goto sys_python
where tar >nul 2>nul
if errorlevel 1 goto sys_python
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_python.ps1" -LauncherDir "%~dp0." -ArchivePath "%TEMP%\wwy_lora_python.tar.gz"
if exist "%PY%" goto run
echo   便携 Python 下载失败，再试一次……
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_python.ps1" -LauncherDir "%~dp0." -ArchivePath "%TEMP%\wwy_lora_python.tar.gz"
if exist "%PY%" goto run

:sys_python
rem 兜底：用系统里的 Python 3.10+
for %%C in (py python) do (
    %%C -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && (
        set "PY=%%C"
        goto run
    )
)
echo.
echo   [x] 没有可用的 Python。请检查网络后重新双击本文件，
echo       或手动安装 Python 3.10+（勾选 Add Python to PATH）：https://www.python.org/downloads/
pause
exit /b 1

:run
"%PY%" -I "%~dp0app\main.py" %*
if errorlevel 1 (
    echo.
    echo   [x] 训练器异常退出，上面是错误信息。
    pause
)
