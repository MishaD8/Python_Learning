class Product:
    def __init__(self, name, price, quantity=0):
        self.name = name
        self.price = price
        self.quantity = quantity

    def calculate_total(self):
        return self.price * self.quantity
    
    def add_inventory(self, amount):
        self.quantity += amount
        return self.quantity
    
    def sell(self, amount):
        if amount <= self.quantity:
            self.quantity -= amount
            return True
        return False
    
    def __str__(self):
        return f"Product: {self.name}, Price: ${self.price}, Available: {self.quantity}"