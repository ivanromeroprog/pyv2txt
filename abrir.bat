@echo off
rem ============================================================================
rem  Lanzador de la interfaz grafica (Windows)
rem  Doble clic en este archivo, o:  abrir.bat
rem
rem  Busca Python (venv local -> "py -3" -> "python"), verifica las
rem  dependencias y abre interfaz.py.
rem ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Transcriptor de audio - Whisper

echo.
echo   Iniciando el Transcriptor de audio...
echo.

rem --- 1) Buscar un interprete de Python ------------------------------------
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if defined PY goto :comprobar

py -3 -c "pass" >nul 2>&1
if not errorlevel 1 (
    set "PY=py -3"
    goto :comprobar
)

python -c "pass" >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
    goto :comprobar
)

echo   [ERROR] No se encontro Python en este equipo.
echo.
echo   Descargalo desde  https://www.python.org/downloads/
echo   e instalalo marcando "Add python.exe to PATH".
echo.
pause
exit /b 1

rem --- 2) Verificar dependencias ---------------------------------------------
:comprobar
echo   Python: %PY%
%PY% -c "import tkinter" >nul 2>&1
if errorlevel 1 goto :falta_tkinter
%PY% -c "import faster_whisper" >nul 2>&1
if errorlevel 1 goto :falta_whisper

rem --- 3) Abrir la interfaz --------------------------------------------------
%PY% interfaz.py
if errorlevel 1 goto :error
endlocal
exit /b 0

rem --- Errores con instrucciones ---------------------------------------------
:falta_tkinter
echo.
echo   [ERROR] Falta tkinter, necesario para la interfaz grafica.
echo   En Windows deberia venir con Python: probá reinstalar Python
echo   con la opcion "tcl/tk and IDLE" marcada.
echo.
pause
exit /b 1

:falta_whisper
echo.
echo   [ERROR] Faltan las dependencias del proyecto.
echo.
echo   Ejecuta este comando y despues volve a abrir este archivo:
echo.
echo       %PY% -m pip install -r requirements.txt
echo.
pause
exit /b 1

:error
echo.
echo   La interfaz se cerro con un error. Lee el mensaje de arriba.
echo.
pause
exit /b 1
