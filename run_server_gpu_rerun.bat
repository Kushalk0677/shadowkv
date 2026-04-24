@echo off
setlocal

set "SSH_TARGET=kushalkhemani@192.168.104.77"
set "SCRIPT_DIR=%~dp0"

echo Running remote refresh and GPU sweep on %SSH_TARGET%...
echo You may be asked for your SSH password.

ssh %SSH_TARGET% "bash -s" < "%SCRIPT_DIR%server_gpu_rerun_remote.sh"

if errorlevel 1 (
  echo.
  echo Remote run failed.
  exit /b 1
)

echo.
echo Remote run completed.
endlocal
