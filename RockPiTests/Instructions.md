# Setup the enviroment
Enviroment is Dp_env

# Setup the connection between both devices

# 1: Vanilla SGD
Set DPSGD to False in the code
run profilingOnLinux.py --test-script set to central

# 2: Central DP
Set DPSGD to False in the code
run profilingOnLinux.py with --test-script set to central

# 3: Vanilla FL
Get the server ip address
copy that to the client and server code
set the flag DPSGD to False
on the server, run server.py and as many clients as you like
on the rockpi, edit the client.py to have the correct ip address
run profilingOnLinux.py --test-script set to central

# 4: FL with DP
Get the server ip address
copy that to the client and server code
set the flag DPSGD to True
on the server, run server.py and as many clients as you like
on the rockpi, edit the client.py to have the correct ip address
run profilingOnLinux.py --test-script set to central