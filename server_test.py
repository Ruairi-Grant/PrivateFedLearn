
import socket                

s = socket.socket()          
print("Socket successfully created")

port = 12345                

s.bind(('192.168.0.11', port))         
print("socket binded to %s" %(port))

s.listen(10)      
print("socket is listening")           

while True: 

   c, addr = s.accept()      
   print('Got connection from', addr) 

   c.send('Thank you for connecting') 

   c.close() 