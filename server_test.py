import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("Socket successfully created")

port = 12345
host = "192.168.0.10"

s.bind((host, port))
print(f"socket binded to {host} on port {port}")


msg_count = 0
while True:

    # Receive the client packet along with the address it is coming from
    message, address = s.recvfrom(1024)
    if message:
        msg_count += 1

    print(f"Received: {message}, count: {msg_count}\r")

    response = "Hello, I am the server".encode("utf-8")

    # Send a reply to the client
    s.sendto(response, address)
