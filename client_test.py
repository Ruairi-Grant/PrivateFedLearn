import socket                

s = socket.socket()          

port = 12345                

s.connect(('192.168.0.11', port)) 

s.send('Hello this is your rock_pi') 

s.close()