@echo off
REM Startup docker containers for local development

echo Starting docker container...

REM Check if the container is already running
set CONTAINER_NAME=tf23studyenv_local

docker ps -q -f name=%CONTAINER_NAME% > nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f %%i in ('docker ps -q -f name^=%CONTAINER_NAME%') do set RUNNING=%%i
)

if defined RUNNING (
    echo Container %CONTAINER_NAME% is already running.
    exit /b 0
)

REM If not running, start a new container
docker run --rm ^
    -p 8888:8888 ^
    -v d:\AI-learning\Deep-learning:/workspace ^
    --name tf23studyenv_local cnn_tf_jupyter:tf29-v01 

