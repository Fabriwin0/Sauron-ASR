@echo off
echo Creating and activating virtual environment...

REM Create virtual environment if it doesn't exist
if not exist "env" (
    echo Creating new virtual environment...
    python -m venv env
)

REM Activate virtual environment
echo Activating virtual environment...
call env\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the integrated system
echo Starting integrated ASR + Chat system...
python run_integrated_system.py

pause