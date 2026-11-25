@echo off
setlocal enableextensions
rem ---------------------------------------------------------------------------------------------
rem  Builds the SeisDesk installer on this Windows PC. Use "build_windows.bat cuda" for CUDA.
rem  Needs Python 3.11, 3.12 or 3.13 (64-bit, from python.org) and an internet connection.
rem  Inno Setup 6 is installed with winget when it is missing.
rem ---------------------------------------------------------------------------------------------
cd /d "%~dp0"
if /i "%~1"=="--help" goto :usage
set "FLAVOUR=CPU"
set "SEISDESK_BUILD_CUDA=0"
set "BUILD_SUFFIX="
set "REQUIREMENTS=packaging\requirements-windows.txt"
if /i "%~1"=="cuda" (
  set "FLAVOUR=CUDA"
  set "SEISDESK_BUILD_CUDA=1"
  set "BUILD_SUFFIX=-CUDA"
  set "REQUIREMENTS=packaging\requirements-windows-cuda.txt"
)
if not "%~1"=="" if /i not "%~1"=="cpu" if /i not "%~1"=="cuda" goto :usage

set "PY="
for %%V in (3.11 3.12 3.13) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.11 - 3.13 was not found.
  echo Install Python 3.11 ^(64-bit^) from https://www.python.org/downloads/windows/ and run this file again.
  goto :fail
)
echo Using %PY%
set "VERSION="
for /f "delims=" %%V in ('%PY% -c "import pathlib, tomllib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])"') do set "VERSION=%%V"
if not defined VERSION goto :fail

echo.
echo [1/6] Build environment (.venv-build-%FLAVOUR%)
if not exist ".venv-build-%FLAVOUR%\Scripts\python.exe" (
  %PY% -m venv .venv-build-%FLAVOUR% || goto :fail
)
set "VPY=.venv-build-%FLAVOUR%\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip || goto :fail
"%VPY%" -m pip install -r "%REQUIREMENTS%" || goto :fail
"%VPY%" -m pip install --no-deps -e . || goto :fail

echo.
echo [2/6] Tests (full report in test_report.txt)
set "QT_QPA_PLATFORM=offscreen"
"%VPY%" -m pytest -q > test_report.txt 2>&1
if errorlevel 1 (
  echo Tests failed - see test_report.txt. Installer creation has stopped.
  goto :fail
) else (
  echo All tests passed.
)

echo.
echo [3/6] Icon and diagnostic screenshots
"%VPY%" packaging\make_icon.py || goto :fail
"%VPY%" tools\v04_screenshots.py seisdesk\docs\images || goto :fail
"%VPY%" tools\v05_screenshots.py seisdesk\docs\images || goto :fail
"%VPY%" tools\v06_screenshots.py seisdesk\docs\images || goto :fail
"%VPY%" tools\v07_screenshots.py seisdesk\docs\images || goto :fail
"%VPY%" tools\v08_screenshots.py seisdesk\docs\images || goto :fail
"%VPY%" tools\v09_screenshots.py seisdesk\docs\images || goto :fail
"%VPY%" tools\v10_screenshots.py seisdesk\docs\images || goto :fail
"%VPY%" tools\v11_screenshots.py seisdesk\docs\images || goto :fail
"%VPY%" tools\v12_screenshots.py seisdesk\docs\images || goto :fail

echo.
echo [4/6] Program folder (PyInstaller)
"%VPY%" -m PyInstaller --noconfirm --clean packaging\seisdesk.spec || goto :fail

echo.
echo [5/6] Self-test of the built program
del "%TEMP%\seisdesk_selftest.txt" 2>nul
start "" /wait "dist\SeisDesk\SeisDesk.exe" --self-test
set "SELFTEST=%errorlevel%"
type "%TEMP%\seisdesk_selftest.txt"
set "QT_QPA_PLATFORM="
if not "%SELFTEST%"=="0" (
  echo The built program failed its self-test; see the text above.
  goto :fail
)
if "%SEISDESK_BUILD_CUDA%"=="1" call :check_cuda
if errorlevel 1 goto :fail

echo.
echo [6/6] Installer (Inno Setup)
call :find_iscc
if not defined ISCC (
  echo Inno Setup 6 was not found; installing it with winget ...
  winget install --id JRSoftware.InnoSetup -e --silent --accept-source-agreements --accept-package-agreements
  call :find_iscc
)
if not defined ISCC (
  echo Inno Setup could not be installed. The program itself is ready: dist\SeisDesk\SeisDesk.exe
  echo Install Inno Setup 6 from https://jrsoftware.org/isdl.php and run this file again for the installer.
  goto :done
)
"%ISCC%" /DAppVersion=%VERSION% "/DBuildSuffix=%BUILD_SUFFIX%" packaging\seisdesk.iss || goto :fail
echo.
echo Done. Installer: dist\SeisDesk-%VERSION%%BUILD_SUFFIX%-setup.exe
goto :done

:check_cuda
del "%TEMP%\seisdesk_gpu_runtime_check.json" 2>nul
start "" /wait "dist\SeisDesk\SeisDesk.exe" --check-gpu-runtime
set "CUDA_CHECK=%errorlevel%"
type "%TEMP%\seisdesk_gpu_runtime_check.json"
exit /b %CUDA_CHECK%

:usage
echo Usage: build_windows.bat [cpu^|cuda]
echo Default: CPU edition. CUDA includes the NVIDIA runtime and compiler components.
echo Both editions require Python 3.11 - 3.13 and Inno Setup 6.
exit /b 0

:find_iscc
set "ISCC="
for %%P in ("%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" "%ProgramFiles%\Inno Setup 6\ISCC.exe" "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe") do (
  if exist "%%~P" set "ISCC=%%~P"
)
exit /b 0

:done
pause
exit /b 0

:fail
echo.
echo BUILD FAILED - see the messages above.
pause
exit /b 1
