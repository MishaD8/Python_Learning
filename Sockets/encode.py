# encoding

# we send bytes

string_var = "something"

print(type(string_var))

string_var = str.encode(string_var)

print(type(string_var))

string_var = string_var.decode()

print(type(string_var))