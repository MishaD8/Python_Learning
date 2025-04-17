# Import the Product class from product.py
from product import Product

# Create some product instances
laptop = Product("Laptop", 899.99, 10)
phone = Product("Smartphone", 499.50, 25)

# Use the class methods
print(laptop)
print(f"Total value of laptop inventory: ${laptop.calculate_total}")

# Add more inventory
laptop.add_inventory(5)
print(f"After restocking: {laptop}")

# Sell some products
if laptop.sell(3):
    print(f"Sold 3 laptops. Remaining: {laptop.quantity}")
else:
    print("Not enough inventory to sell")

# Try all the operations with the phone
print("\n" + str(phone))
print(f"Total value of phone inventory: ${phone.calculate_total}")
phone.add_inventory(10)
print(f"After restocking: {phone}")
if phone.sell(15):
    print(f"Sold 15 phones. Remaining: {phone.quantity}")

# This example demonstrates:
# - Creating a class with attributes and methods
# - Importing that class from another file
# - Creating instances of the class
# - Using various methods of the class
