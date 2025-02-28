@echo off
setlocal enabledelayedexpansion

echo [%date% %time%] Starting setup for Crypto Wallet ^& Trend Analysis application...

:: Check for development mode
set DEV_MODE=0
if "%1"=="dev" (
    set DEV_MODE=1
    echo [%date% %time%] Setting up in development mode
) else (
    echo [%date% %time%] Setting up in standard mode (use 'setup.bat dev' for development mode)
)

:: Check Python version
echo [%date% %time%] Checking Python version...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%date% %time%] [31mX Python not found. Please install Python 3.9 or higher.[0m
    exit /b 1
)

:: Get Python version
for /f "tokens=2 delims=." %%a in ('python -c "import sys; print(sys.version.split()[0])"') do (
    set PYTHON_VERSION=%%a
)
echo [%date% %time%] Found Python version: %PYTHON_VERSION%

:: Check if Python is at least 3.9
if %PYTHON_VERSION% LSS 9 (
    echo [%date% %time%] [33m! Python version %PYTHON_VERSION% detected. Recommended version is 3.9 or higher.[0m
    set /p CONTINUE="Continue anyway? (y/n) "
    if /i "!CONTINUE!"=="y" (
        echo Continuing with existing Python version...
    ) else (
        exit /b 1
    )
) else (
    echo [%date% %time%] [32m✓ Python version is compatible[0m
)

:: Setup virtual environment
echo [%date% %time%] Setting up virtual environment...
if exist ".venv" (
    echo [%date% %time%] [33m! Virtual environment already exists.[0m
    set /p RECREATE="Do you want to recreate it? (y/n) "
    if /i "!RECREATE!"=="y" (
        rmdir /s /q .venv
    ) else (
        echo [%date% %time%] Using existing virtual environment.
        goto :activate_venv
    )
)

:: Create virtual environment
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo [%date% %time%] [31mX Failed to create virtual environment. Make sure venv module is available.[0m
    exit /b 1
)
echo [%date% %time%] [32m✓ Virtual environment created in .venv directory[0m

:activate_venv
:: Activate virtual environment
echo [%date% %time%] Activating virtual environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [%date% %time%] [32m✓ Virtual environment activated[0m
) else (
    echo [%date% %time%] [31mX Virtual environment activation script not found. Setup may be incomplete.[0m
    exit /b 1
)

:: Upgrade pip
echo [%date% %time%] Upgrading pip...
python -m pip install --upgrade pip
echo [%date% %time%] [32m✓ Pip upgraded to latest version[0m

:: Install dependencies
echo [%date% %time%] Installing dependencies...
if not exist "requirements.txt" (
    echo [%date% %time%] [31mX requirements.txt not found. Are you in the right directory?[0m
    exit /b 1
)

pip install -r requirements.txt
echo [%date% %time%] [32m✓ Dependencies installed successfully[0m

:: Install development dependencies if in dev mode
if %DEV_MODE%==1 (
    echo [%date% %time%] Installing development dependencies...
    if exist "requirements-dev.txt" (
        pip install -r requirements-dev.txt
        echo [%date% %time%] [32m✓ Development dependencies installed successfully[0m
    ) else (
        echo [%date% %time%] No development dependencies file found, installing common development packages...
        pip install pytest pytest-cov black flake8 mypy
        echo [%date% %time%] [32m✓ Common development packages installed[0m
    )
)

:: Setup environment
echo [%date% %time%] Setting up environment variables...
if not exist ".env.example" (
    echo [%date% %time%] [31mX .env.example not found. Are you in the right directory?[0m
    exit /b 1
)

if not exist ".env" (
    copy .env.example .env
    echo [%date% %time%] [32m✓ Created .env file from .env.example[0m
    echo [%date% %time%] [33m! Please edit the .env file with your actual API keys and configuration.[0m
) else (
    echo [%date% %time%] [33m! .env file already exists. Make sure it contains all necessary environment variables.[0m
)

:: Initialize database
echo [%date% %time%] Initializing database...
if not exist "init_db.py" (
    echo [%date% %time%] [31mX init_db.py not found. Are you in the right directory?[0m
    exit /b 1
)

python init_db.py
echo [%date% %time%] [32m✓ Database initialized successfully[0m

:: Create convenience scripts
echo [%date% %time%] Creating convenience scripts...

:: Create run script
echo @echo off > scripts\run.bat
echo if exist ".venv\Scripts\activate.bat" ( >> scripts\run.bat
echo     call .venv\Scripts\activate.bat >> scripts\run.bat
echo ) else ( >> scripts\run.bat
echo     echo Virtual environment not found. Run setup.bat first. >> scripts\run.bat
echo     exit /b 1 >> scripts\run.bat
echo ) >> scripts\run.bat
echo. >> scripts\run.bat
echo streamlit run app.py %* >> scripts\run.bat

:: Create test script
echo @echo off > scripts\test.bat
echo if exist ".venv\Scripts\activate.bat" ( >> scripts\test.bat
echo     call .venv\Scripts\activate.bat >> scripts\test.bat
echo ) else ( >> scripts\test.bat
echo     echo Virtual environment not found. Run setup.bat first. >> scripts\test.bat
echo     exit /b 1 >> scripts\test.bat
echo ) >> scripts\test.bat
echo. >> scripts\test.bat
echo pytest %* >> scripts\test.bat

:: Create lint script
echo @echo off > scripts\lint.bat
echo if exist ".venv\Scripts\activate.bat" ( >> scripts\lint.bat
echo     call .venv\Scripts\activate.bat >> scripts\lint.bat
echo ) else ( >> scripts\lint.bat
echo     echo Virtual environment not found. Run setup.bat first. >> scripts\lint.bat
echo     exit /b 1 >> scripts\lint.bat
echo ) >> scripts\lint.bat
echo. >> scripts\lint.bat
echo echo Running flake8... >> scripts\lint.bat
echo flake8 src app.py >> scripts\lint.bat
echo echo Running mypy... >> scripts\lint.bat
echo mypy src app.py >> scripts\lint.bat
echo echo Running black (check only)... >> scripts\lint.bat
echo black --check src app.py >> scripts\lint.bat

:: Create format script
echo @echo off > scripts\format.bat
echo if exist ".venv\Scripts\activate.bat" ( >> scripts\format.bat
echo     call .venv\Scripts\activate.bat >> scripts\format.bat
echo ) else ( >> scripts\format.bat
echo     echo Virtual environment not found. Run setup.bat first. >> scripts\format.bat
echo     exit /b 1 >> scripts\format.bat
echo ) >> scripts\format.bat
echo. >> scripts\format.bat
echo black src app.py >> scripts\format.bat

echo [%date% %time%] [32m✓ Convenience scripts created in scripts directory[0m

:: Final success message
echo.
echo [%date% %time%] [32m✓ Setup completed successfully![0m
echo.
echo To activate the virtual environment in the future, run:
echo   .venv\Scripts\activate.bat
echo.
echo To run the application:
echo   scripts\run.bat
echo.

if %DEV_MODE%==1 (
    echo Development tools:
    echo   Run tests:        scripts\test.bat
    echo   Lint code:        scripts\lint.bat
    echo   Format code:      scripts\format.bat
    echo.
)

echo Remember to edit the .env file with your actual API keys and configuration.

endlocal