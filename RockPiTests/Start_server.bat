@echo off
title Server
cd "C:\git_repos\Thesis\RockPiTests"
rem Activate the flower env
call DP_env\Scripts\activate.bat
rem Start server.py
rem python fl_server.py --rounds 3 --min_num_clients 3  --sample_fraction 1.0 --server_address "127.0.0.1:65432" --local_epochs 20
python -i server.py --num-rounds 4 --num-clients 3  --server-address "192.168.0.10:8080"