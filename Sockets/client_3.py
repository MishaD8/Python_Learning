import socket

def start_client():
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
        print("Type messages to send to server. Type 'quit' to exit.")
        
        # Infinite loop for continuous communication
        while True:
            try:
                # Get message from user
                message = input("Enter message: ")
                
                if message.lower() == 'quit':
                    print("Disconnecting...")
                    c.send(message.encode('utf-8'))
                    break
                
                # Send message to server
                c.send(message.encode('utf-8'))
                print(f"Sent: {message}")
                
                # Receive response from server
                response = c.recv(1024).decode('utf-8')
                print(f"Server response: {response}")
                
            except KeyboardInterrupt:
                print("\nDisconnecting...")
                break
            except ConnectionResetError:
                print("Server disconnected")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
                
    except ConnectionRefusedError:
        print("Error: Could not connect to server. Make sure the server is running.")
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        if 'c' in locals():
            c.close()
            print("Connection closed.")

if __name__ == "__main__":
    start_client()