import socket                

s = socket.socket()          

port = 12345                

s.connect(('192.168.0.10', port)) 

s.send('Hello this is your rock_pi'.encode('utf-8')) 

data = s.recv(1024)

print(f"Received {data!r}")
s.close()
