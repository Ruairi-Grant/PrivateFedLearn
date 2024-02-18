@echo off
title Clients
set SCRIPT_DIR="C:\git_repos\Thesis"
rem Number of times to run the script
set NUM_RUNS=3

rem Activate the flower env
call flowerEnv\Scripts\activate.bat
rem Number of times to run the script
set NUM_RUNS=3

rem Run the Python script multiple times in different terminals
for /l %%i in (1,1,%NUM_RUNS%) do (
    start cmd /k "cd %SCRIPT_DIR% && call flowerEnv\Scripts\activate.bat && python fl_client.py --cid=%%i --server_address "127.0.0.1:65432""
)