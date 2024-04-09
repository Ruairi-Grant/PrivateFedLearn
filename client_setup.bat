set cid=%1
set SCRIPT_DIR=C:\git_repos\Thesis

title Client %1
cd %SCRIPT_DIR%
call flwr_fhe\Scripts\activate.bat
rem python fl_client_dp.py  --server_address "127.0.0.1:65432" --cid %1
python mnist_client.py  --server_address "127.0.0.1:65432" --cid %1