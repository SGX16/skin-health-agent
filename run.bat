@echo off
chcp 65001 >nul
cd /d %~dp0

rem 优先使用项目内的虚拟环境，否则回退到系统 PATH 中的 python
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if exist "%CD%\models\qwen3-vl-2b" (
    "%PY%" app.py --model ./models/qwen3-vl-2b
) else (
    "%PY%" app.py --model Qwen/Qwen3-VL-2B-Instruct
)
pause
