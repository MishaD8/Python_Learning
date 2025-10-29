ip_address = "54.205.228.2"
port = 8000
is_secure = True
response_time = 0.038

print(f"Address: {ip_address}")
print(f"Port: {port}")
print(f"Secure: {is_secure}")
print(f"Time: {response_time}")

print(type(ip_address))
print(type(port))
print(type(is_secure))
print(type(response_time))

print("\n--- Bonus ---")
full_address = ip_address + ":" + str(port)
print(f"Full address: {full_address}")