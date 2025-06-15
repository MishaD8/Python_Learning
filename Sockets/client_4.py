import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# IP and host that we want to connect to

host = '127.0.0.1'
port = 9002

s.connect((host, port))

# we are now connected to the server

# client receiving the data

while True:
    msg = s.recv(1024).decode()
    if not msg:
        break
    print(msg)
    data = input('Enter a msg   ')
    s.sendall(str.encode(data))

s.close()