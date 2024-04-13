set cid=%1
set SCRIPT_DIR=C:\git_repos\Thesis\RockPiTests

title Client %1
cd %SCRIPT_DIR%
call DP_env\Scripts\activate.bat
python client.py  --server-address "192.168.0.10:8080" --partition=%1 --dpsgd=True