import bcrypt

pw = "Password123"

hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

print(hashed)