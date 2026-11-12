@echo off
setlocal
if "%~1"=="" exit /b 1
set "LOG=%TEMP%\am_fastview_log.txt"

> "%LOG%" (
  echo [%date% %time%] File "%~1"
)

pushd "%~dp1"
"C:\Users\simon\anaconda3\envs\vessel_growth\python.exe" -m pymira.fast_view "%~1" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
popd

if %RC% neq 0 (
  echo Failed with exit code %RC% >> "%LOG%"
  start notepad "%LOG%"
)
exit /b %RC%
