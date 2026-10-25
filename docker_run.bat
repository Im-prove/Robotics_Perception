@echo off
REM =====================================================================
REM  Autonomous Robotics Perception Pipeline - Docker Build & Run Script
REM =====================================================================

echo [*] Checking Docker daemon status...
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] ERROR: Docker Desktop is not running.
    echo Please start Docker Desktop from your Start Menu and try again.
    pause
    exit /b 1
)

echo [*] Building Docker image: eric-robotics/perception-pipeline:1.0.0...
docker build -t eric-robotics/perception-pipeline:1.0.0 .

if %ERRORLEVEL% NEQ 0 (
    echo [!] Build failed.
    pause
    exit /b 1
)

echo.
echo [*] Running Perception Pipeline Container...
docker run --rm ^
  -v "%cd%\data:/app/data:ro" ^
  -v "%cd%\output:/app/output:rw" ^
  -v "%cd%\weights:/app/weights:ro" ^
  eric-robotics/perception-pipeline:1.0.0 ^
  python deploy_node.py --source data/bdd100k_sample/images --output output/docker_run

echo.
echo [OK] Execution finished! Check results in: output\docker_run
pause
