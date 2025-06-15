import socket

def start_server():
    # Server password
    SERVER_PASSWORD = "cybersecurity123"
    
    try:
        # Create socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Server details
        host = '127.0.0.1'
        port = 9001
        
        # Bind and listen
        s.bind((host, port))
        print(f"Password-protected server listening on {host}:{port}")
        print(f"Server password: {SERVER_PASSWORD}")
        s.listen(5)
        
        print("Server started. Press Ctrl+C to stop.")
        
        # Infinite loop to handle multiple clients
        while True:
            try:
                print("\nWaiting for client connection...")
                conn, addr = s.accept()
                print(f'Connection attempt from {addr}')
                
                # First, get password from client
                try:
                    password = conn.recv(1024).decode('utf-8')
                    print(f"Password received from {addr}: {password}")
                    
                    # Check password
                    if password != SERVER_PASSWORD:
                        print(f"Access denied for {addr} - wrong password")
                        conn.send("access denied".encode('utf-8'))
                        conn.close()
                        continue
                    
                    # Password correct - send confirmation
                    conn.send("access granted".encode('utf-8'))
                    print(f"Access granted to {addr}")
                    
                    # Start chat session
                    print(f"Chat session started with {addr}")
                    
                    while True:
                        try:
                            # Receive message from client
                            client_msg = conn.recv(1024).decode('utf-8')
                            
                            if not client_msg:
                                print(f"Client {addr} disconnected")
                                break
                            
                            if client_msg.lower() == 'end':
                                print(f"Client {addr} ended the session")
                                break
                            
                            print(f"CLIENT>> {client_msg}")
                            
                            # Get server response
                            server_response = input("SERVER>> ")
                            
                            # Send response to client
                            conn.send(server_response.encode('utf-8'))
                            
                        except ConnectionResetError:
                            print(f"Client {addr} disconnected unexpectedly")
                            break
                        except Exception as e:
                            print(f"Error in chat with {addr}: {e}")
                            break
                    
                except Exception as e:
                    print(f"Error during password check with {addr}: {e}")
                
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