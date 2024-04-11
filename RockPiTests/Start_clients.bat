@echo off
title Clients

rem Number of times to run the script
set NUM_RUNS=1

rem Run the Python script multiple times in different terminals
for /l %%i in (1,1,%NUM_RUNS%) do (
    tasklist /FI "WINDOWTITLE eq Client %%i" | findstr /C:"cmd.exe" > nul
    if not errorlevel 1 (
        taskkill /FI "WINDOWTITLE eq Client %%i"
    )
    start cmd /k call client_setup.bat %%i
)




