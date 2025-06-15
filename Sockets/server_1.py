import socket

try:
    # Create socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(type(s))
    
    # Server details
    host = '127.0.0.1'
    port = 9001
    
    # Bind and listen
    s.bind((host, port))
    print(f"Server listening on {host}:{port}")
    
    s.listen()
    print("Waiting for client connection...")
    
    # Accept connection
    conn, addr = s.accept()
    print(f'Connection received from {addr}')
    
    # Receive data from client
    data = conn.recv(1024).decode('utf-8')
    print(f"Received from client: {data}")
    
    # Send response back to client
    response = "Hello from server!"
    conn.send(response.encode('utf-8'))
    print(f"Sent to client: {response}")
    
except OSError as e:
    print(f"Error: {e}")
    if "10048" in str(e):
        print("Port is already in use. Try a different port or wait a moment.")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
        print("Client connection closed.")
    if 's' in locals():
        s.close()
        print("Server socket closed.")