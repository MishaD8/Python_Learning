import socket
import getpass

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
        
        # Get password from user (hidden input)
        password = getpass.getpass("Enter server password: ")
        
        # Send password to server
        c.send(password.encode('utf-8'))
        
        # Wait for server response
        auth_response = c.recv(1024).decode('utf-8')
        
        if auth_response == "access denied":
            print("ACCESS DENIED - Wrong password!")
            return
        elif auth_response == "access granted":
            print("ACCESS GRANTED - Welcome to the chat!")
            print("Type 'end' to quit the chat session.")
            print("-" * 40)
        
        # Chat loop
        while True:
            try:
                # Get message from user
                message = input("CLIENT>> ")
                
                # Send message to server
                c.send(message.encode('utf-8'))
                
                # Check if user wants to end
                if message.lower() == 'end':
                    print("Ending chat session...")
                    break
                
                # Receive response from server
                response = c.recv(1024).decode('utf-8')
                print(f"SERVER>> {response}")
                
            except KeyboardInterrupt:
                print("\nEnding chat session...")
                c.send("end".encode('utf-8'))
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