@echo off
REM ===== deepvoice-bench: one-click setup and check (Windows) =====
REM Double-click this file. All output is saved in the logs folder.
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
if not exist setup_weights.py (
  echo Unpacking deepvoice-bench.zip ...
  tar -xf deepvoice-bench.zip
)
if not exist logs mkdir logs
set "DLOG=logs\00_diagnose.txt"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo [1/6] Checking this computer...
echo ===== %date% %time% > "%DLOG%"
echo ===== OS >> "%DLOG%"
ver >> "%DLOG%" 2>&1
echo ===== py launcher >> "%DLOG%"
py -0p >> "%DLOG%" 2>&1
echo ===== python on PATH >> "%DLOG%"
where python >> "%DLOG%" 2>&1
echo ===== nvidia-smi >> "%DLOG%"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> "%DLOG%" 2>&1
echo ===== git >> "%DLOG%"
git --version >> "%DLOG%" 2>&1
echo ===== ffmpeg >> "%DLOG%"
where ffmpeg >> "%DLOG%" 2>&1
echo ===== disk and cpu >> "%DLOG%"
powershell -NoProfile -Command "Get-PSDrive C | Format-List Used,Free; Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores; Get-CimInstance Win32_VideoController | Format-List Name,AdapterRAM,DriverVersion; 'RAM_GB=' + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)" >> "%DLOG%" 2>&1

echo [2/6] Finding Python 3.10-3.14...
set "PY="
for %%v in (3.12 3.11 3.13 3.14 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,14)" >nul 2>&1 && set "PY=python"
)
echo ===== selected python: !PY! >> "%DLOG%"
if not defined PY (
  echo.
  echo  Python 3.10-3.14 not found. Install Python 3.12 from python.org
  echo  ^(check "Add python.exe to PATH"^), then run this file again.
  goto :end
)

echo [3/6] Creating virtual environment .venv ...
if not exist ".venv\Scripts\python.exe" (
  !PY! -m venv .venv >> "%DLOG%" 2>&1
)
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo  Failed to create .venv - see logs\00_diagnose.txt
  goto :end
)

echo [4/6] Installing packages ^(first time: 10-30 min^)...
"%VPY%" run_logged.py logs\01_pip.txt "%VPY%" -m pip install --upgrade pip
set "TORCH_INDEX="
nvidia-smi >nul 2>&1 && set "TORCH_INDEX=--index-url https://download.pytorch.org/whl/cu128"
"%VPY%" -c "import torch" >nul 2>&1 || "%VPY%" run_logged.py logs\01_pip.txt "%VPY%" -m pip install torch torchaudio !TORCH_INDEX!
"%VPY%" run_logged.py logs\01_pip.txt "%VPY%" -m pip install -r requirements-bench.txt
"%VPY%" run_logged.py logs\01_pip.txt "%VPY%" -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', '| transformers', transformers.__version__)"

echo [5/6] Downloading model weights ^(about 2.5GB^)...
"%VPY%" run_logged.py logs\02_setup_weights.txt setup_weights.py

echo [6/6] Running checks...
"%VPY%" run_logged.py logs\03_check_env.txt check_env.py
"%VPY%" run_logged.py logs\04_sanity_toy.txt sanity_toy.py

:end
echo.
echo  Finished. Logs are in the logs folder. Tell Claude "done".
pause
