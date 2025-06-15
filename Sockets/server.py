
import socket

# socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(type(s))

host = '127.0.0.1'
port = 9002

# bind

s.bind((host, port)) # should be over 1024

# listen

s.listen() # now waiting

# accept

conn, addr = s.accept() # not do this until someone is trying to connect a client
print(f'Connection receive from {addr}')