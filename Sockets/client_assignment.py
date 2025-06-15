import socket
import getpass


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# IP and host that we want to connect to

host = '127.0.0.1'
port = 9002

s.connect((host, port))

# get password from user (hidden input)

password = getpass.getpass("Enter server password:   ")

# we are now connected to the server

# client receiving the data

while True:
    msg = s.recv(1024).decode()
    if not msg:
        break
    print(msg)
    data = input('Enter a msg   ')
    s.sendall(str.encode(data))
    if msg.lower() == 'end':
        print("Ending chat session...")
        break

s.close()