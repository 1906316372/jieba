@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHON=%~dp0.conda\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [ERROR] conda Python not found:
    echo   %PYTHON%
    echo Make sure the project .conda environment exists.
    pause
    exit /b 1
)

:menu
cls
echo ========================================
echo   SF6 Script Launcher
echo   Python: .conda\Scripts\python.exe
echo ========================================
echo.

set "COUNT=0"
for /f "delims=" %%F in ('dir /b /a:-d /o:n "*.py" 2^>nul') do (
    set /a COUNT+=1
    set "SCRIPT_!COUNT!=%%F"
    echo   !COUNT!. %%F
)

if %COUNT% equ 0 (
    echo [ERROR] No .py files in this folder.
    pause
    exit /b 1
)

echo.
echo   0. Exit
echo ========================================
set /p "CHOICE=Enter number and press Enter: "

if "%CHOICE%"=="" goto menu
if "%CHOICE%"=="0" exit /b 0

set /a "NUM=CHOICE" 2>nul
if "%NUM%"=="" goto invalid
if %NUM% lss 1 goto invalid
if %NUM% gtr %COUNT% goto invalid

call set "TARGET=%%SCRIPT_%NUM%%%"
if not defined TARGET goto invalid
if not exist "%TARGET%" goto invalid

echo.
echo ^>^>^> Starting: %TARGET%
echo ^>^>^> Using: %PYTHON%
echo.
"%PYTHON%" "%TARGET%"
set "EXITCODE=%ERRORLEVEL%"
echo.
echo ^>^>^> Finished (exit code %EXITCODE%)
pause
goto menu

:invalid
echo.
echo [ERROR] Invalid number, try again.
timeout /t 2 >nul
goto menu