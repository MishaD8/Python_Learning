import socket

def start_server():
    try:
        # Create socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Server details
        host = '127.0.0.1'
        port = 9001
        
        # Bind and listen
        s.bind((host, port))
        print(f"Server listening on {host}:{port}")
        s.listen(5)  # Allow up to 5 pending connections
        
        print("Server started. Press Ctrl+C to stop.")
        
        # Infinite loop to handle multiple clients
        while True:
            try:
                print("\nWaiting for client connection...")
                conn, addr = s.accept()
                print(f'Connection established with {addr}')
                
                # Handle this client
                while True:
                    try:
                        # Receive data from client
                        data = conn.recv(1024).decode('utf-8')
                        
                        if not data or data.lower() == 'quit':
                            print(f"Client {addr} disconnected")
                            break
                        
                        print(f"Received from {addr}: {data}")
                        
                        # Send response back to client
                        response = f"Server received: {data}"
                        conn.send(response.encode('utf-8'))
                        
                    except ConnectionResetError:
                        print(f"Client {addr} disconnected unexpectedly")
                        break
                    except Exception as e:
                        print(f"Error handling client {addr}: {e}")
                        break
                
                conn.close()
                print(f"Connection with {addr} closed")
                
            except Exception as e:
                print(f"Error accepting connection: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\nServer shutting down...")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        if 's' in locals():
            s.close()
            print("Server socket closed")

if __name__ == "__main__":
    start_server()