@echo off
REM ===== Build Korean evaluation data + score with 3 detectors =====
REM Requires setup_and_check.bat to have been run once.
REM Takes about 1.5-2.5 hours the first time. Safe to re-run: finished parts are skipped.
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
if exist update_ko.zip (
  echo Unpacking update_ko.zip ...
  tar -xf update_ko.zip && del update_ko.zip
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set COQUI_TOS_AGREED=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
if not exist logs mkdir logs
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo  .venv not found. Run setup_and_check.bat first.
  goto :end
)
"%VPY%" -c "import sys; open('logs/base_python.txt','w').write(sys._base_executable)"
set /p BASEPY=<logs\base_python.txt
echo Base Python: !BASEPY!

echo [1/5] Korean real speech: Zeroth-Korean test set ...
"%VPY%" run_logged.py logs\10_pip_main.txt "%VPY%" -m pip install -r requirements-bench.txt
"%VPY%" run_logged.py logs\11_prepare_ko.txt 01_prepare_ko.py --n 200 --telephony g711
if not exist data\real_ko\tts_meta.csv (
  echo  Korean data preparation failed - see logs\11_prepare_ko.txt
  goto :end
)

echo [2/5] Codec/vocoder resynthesis ...
call :mkenv codec
if exist ".venv_codec\Scripts\python.exe" ".venv_codec\Scripts\python.exe" run_logged.py logs\21_resynth.txt 02_resynth.py --real data/real_ko/manifest.csv --out data/fake_ko --telephony g711 --no_real_telephony

echo [3/5] Voice-cloning TTS: xtts, chatterbox, omnivoice ...
for %%s in (xtts chatterbox omnivoice) do (
  echo   - %%s
  call :mkenv %%s
  if "%%s"=="xtts" ".venv_xtts\Scripts\python.exe" run_logged.py logs\20_env_xtts.txt ".venv_xtts\Scripts\python.exe" -m pip install --no-deps torchcodec
  if exist ".venv_%%s\Scripts\python.exe" ".venv_%%s\Scripts\python.exe" run_logged.py logs\3_tts_%%s.txt 02b_tts.py --system %%s --telephony g711
)

echo [4/5] Scoring with 3 detectors ...
set "M=--manifest data/real_ko/manifest.csv"
if exist data\fake_ko\manifest.csv set "M=!M! --manifest data/fake_ko/manifest.csv"
for %%s in (xtts chatterbox omnivoice) do (
  if exist data\fake_ko_tts\%%s\manifest.csv set "M=!M! --manifest data/fake_ko_tts/%%s/manifest.csv"
)
"%VPY%" run_logged.py logs\40_score.txt 03_score.py !M! --out results/scores_ko.csv

echo [5/5] Analysis ...
"%VPY%" run_logged.py logs\41_analyze.txt 04_analyze.py --scores results/scores_ko.csv --out results/ko
"%VPY%" run_logged.py logs\42_compare.txt 05_compare.py --scores results/scores_ko.csv --out results/ko
goto :end

:mkenv
set "E=.venv_%~1"
if not exist "!E!\Scripts\python.exe" (
  echo     creating !E! ...
  "!BASEPY!" -m venv "!E!" >> logs\20_env_%~1.txt 2>&1
)
if not exist "!E!\Scripts\python.exe" (
  echo     failed to create !E! - see logs\20_env_%~1.txt
  exit /b 1
)
"!E!\Scripts\python.exe" run_logged.py logs\20_env_%~1.txt "!E!\Scripts\python.exe" -m pip install --upgrade pip
"!E!\Scripts\python.exe" -c "import torch" >nul 2>&1 || "!E!\Scripts\python.exe" run_logged.py logs\20_env_%~1.txt "!E!\Scripts\python.exe" -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"!E!\Scripts\python.exe" run_logged.py logs\20_env_%~1.txt "!E!\Scripts\python.exe" -m pip install -r req-gen-%~1.txt -c constraints-gen.txt
exit /b 0

:end
echo.
echo  Finished. Logs are in the logs folder. Tell Claude "done".
pause
