1. Variables and Data Types
Theory
In Python, a variable is a container that stores data. You don't need to declare the type explicitly—Python figures it out automatically (this is called "dynamic typing"). Main data types:
int - whole numbers (e.g., 42, -10)
float - decimal numbers (e.g., 3.14, -0.5)
str - text/strings (e.g., "hello", 'AWS')
bool - True or False
Why this matters for cloud/security:
You'll store IP addresses, instance IDs, log entries, port numbers, etc.
Knowing types helps you avoid errors when parsing API responses or config files
Code Example
# Variables - no need to declare type
server_name = "web-server-01"           # string
port = 443                               # integer
is_running = True                        # boolean
cpu_usage = 87.5                        # float

# Print variables
print(f"Server: {server_name}")
print(f"Port: {port}")
print(f"Running: {is_running}")
print(f"CPU Usage: {cpu_usage}%")

# Check the type of a variable
print(type(server_name))  # <class 'str'>
print(type(port))         # <class 'int'>
Try running this in your terminal:
python3 -c "
server_name = 'web-server-01'
port = 443
print(f'Server: {server_name} on port {port}')
"
Type Casting
Sometimes you need to convert between types:
# String to integer
port_string = "8080"
port_number = int(port_string)
print(port_number + 80)  # 8160

# Integer to string
instance_count = 5
message = "Running " + str(instance_count) + " instances"
print(message)  # Running 5 instances

# String to float
cpu = "75.3"
cpu_float = float(cpu)
print(cpu_float + 10.0)  # 85.3
Real-World Cloud Example
# AWS EC2 instance info
instance_id = "i-0abc123def456"
instance_type = "t3.medium"
running = True
monthly_cost = 30.50

# Security check: flag high-cost instances
if monthly_cost > 50.0:
    alert = f"⚠️ High cost instance: {instance_id}"
else:
    alert = f"✓ Cost acceptable: {instance_id}"

print(alert)
print(f"Type: {instance_type}, Running: {running}, Cost: ${monthly_cost}")

Your Practice Exercise

Write a Python script that:

# Creates 4 variables:

ip_address (string) - any IP like "192.168.1.10"
port (integer) - like 22 for SSH
is_secure (boolean) - True if using encrypted connection
response_time (float) - like 0.045 seconds

Print them using f-strings in a readable format

Use type() to confirm each variable's type

Bonus: Convert port to a string and concatenate it with the IP address like "192.168.1.10:22"

Save it as basics_exercise1.py and run it. Let me know when you're done or if you have questions!


ip_address = "54.205.228.2"
port = 8000
is_secure = True
response_time = 0.038

print(f"Address: {ip_address}")
print(f"Port: {port}")
print(f"Secure: {is_secure}")
print(f"Time: {response_time}")

# Add these lines to check types
print("\n--- Variable Types ---")
print(f"ip_address type: {type(ip_address)}")
print(f"port type: {type(port)}")
print(f"is_secure type: {type(is_secure)}")
print(f"response_time type: {type(response_time)}")

# Bonus: Concatenate IP + port as a string
print("\n--- Bonus ---")
full_address = ip_address + ":" + str(port)
print(f"Full address: {full_address}")
Add those sections to your file and run it again with python3 Basics/variable.py. What you'll learn:
type() shows you the data type of any variable
When concatenating strings, you need str() to convert numbers to strings
This is useful in cloud work when building URLs, connection strings, or logging

# Review & Key Takeaways

# What you did well:

Used meaningful variable names (ip_address, not just x)
Correctly converted port to string using str(port) for concatenation
Clean, readable output formatting

# Why this matters in cloud/security: This is exactly how you'd build connection strings for:
SSH connections: ssh user@54.205.228.2:8000
API endpoints: https://54.205.228.2:8000/api/health
Logging server info: "Connecting to 54.205.228.2:8000"

# Next Step: Input/Output and F-strings
Ready to move on? The next lesson will cover:
Taking user input with input()
Advanced f-string formatting
String methods for cleaning/parsing data (useful for log analysis)
