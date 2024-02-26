set cid=%1
set SCRIPT_DIR=C:\git_repos\Thesis

title Client %1
cd %SCRIPT_DIR%
call flowerEnv\Scripts\activate.bat
python fl_client_dp.py  --server_address "127.0.0.1:65432" --cid %1