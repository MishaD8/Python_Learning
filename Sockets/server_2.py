import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# host and port

host = '127.0.0.1'
port = 9002

# set up to take connections

s.bind((host, port))
s.listen()
client, addr = s.accept()

# we now have someone connected

print(f'Connection receive from {addr}')

# if server is sending

data = "This is a test, thank you for connecting to the server"
client.sendall(str.encode(data))
client.close()
print(f'Connection with {addr} has been closed')