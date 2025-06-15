import socket

try:
    # Create socket
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Server details
    host = '127.0.0.1'
    port = 9001
    
    # Connect to server
    print(f"Connecting to server at {host}:{port}...")
    c.connect((host, port))
    print("Connected to server!")
    
    # Send data to server
    message = "Hello from client!"
    c.send(message.encode('utf-8'))
    print(f"Sent: {message}")
    
    # Receive response from server
    response = c.recv(1024).decode('utf-8')
    print(f"Received: {response}")
    
except ConnectionRefusedError:
    print("Error: Could not connect to server. Make sure the server is running.")
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'c' in locals():
        c.close()
        print("Connection closed.")