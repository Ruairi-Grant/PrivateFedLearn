# Setup the enviroment
Enviroment is Dp_env

# Setup the connection between both devices
connect both on the same local network and note the ip address of the server computer

# 1: Vanilla SGD
run python profilingOnLinux.py --test-script="central" --dpsgd=False

# 2: Central DP
run python profilingOnLinux.py --test-script="central" --dpsgd=True

# 3: Vanilla FL
on the server, 
set your ip address in the start_server.bat and client_setup.bat
run both bat files
on the rockpi 
run python profilingOnLinux.py --test-script="fl_client" --dpsgd=False --server-address=<IP-ADDRESS> --num-clients=<NUM-CLIENTS>

# 4: FL with DP
on the server, 
set your ip address in the start_server.bat and client_setup.bat
in client_setup set dpsgd to true
run start_server.bat and start_client.bat
on the rockpi 
run python profilingOnLinux.py --test-script="fl_client" --dpsgd=True --server-address=<IP-ADDRESS> --num-clients=<NUM-CLIENTS>