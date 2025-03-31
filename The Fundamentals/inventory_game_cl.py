# Create an inventory tracking system that has all of the following features:

# Each item will have a name, quantity, and description as a dictionary.

# All dictionaries will be saved in a master inventory list.

# A function that adds something to that list (creates the dictionary and adds to the list).

# A function that displays the inventory in a readable format (i.e. do not just print out 
# the list or dictionary, separate them out).

# A function that will change the quantity of an item (this is difficult to do and many 
# ways to do it).

# A function that removes an item from the list. Research required :)


def add_item(inventory, name, quantity, description):
    """
    Add a new item to the inventory.
    
    Args:
        inventory (list): The master inventory list
        name (str): Name of the item
        quantity (int): Quantity of the item
        description (str): Description of the item
    
    Returns:
        bool: True if added successfully, False if item already exists
    """
    # Check if item already exists (avoid duplicates)
    for item in inventory:
        if item['name'].lower() == name.lower():
            print(f"Item '{name}' already exists in inventory.")
            return False
    
    # Create new item dictionary
    new_item = {
        'name': name,
        'quantity': quantity,
        'description': description
    }
    
    # Add to inventory
    inventory.append(new_item)
    print(f"Added '{name}' to inventory.")
    return True


def display_inventory(inventory):
    """
    Display the inventory in a readable format.
    
    Args:
        inventory (list): The master inventory list
    """
    if not inventory:
        print("Inventory is empty.")
        return
    
    print("\n===== INVENTORY =====")
    print(f"Total items: {len(inventory)}")
    print("=====================")
    
    for i, item in enumerate(inventory, 1):
        print(f"Item #{i}:")
        print(f"  Name: {item['name']}")
        print(f"  Quantity: {item['quantity']}")
        print(f"  Description: {item['description']}")
        print("---------------------")


def update_quantity(inventory, name, new_quantity):
    """
    Update the quantity of an existing item.
    
    Args:
        inventory (list): The master inventory list
        name (str): Name of the item to update
        new_quantity (int): New quantity value
    
    Returns:
        bool: True if updated successfully, False if item not found
    """
    for item in inventory:
        if item['name'].lower() == name.lower():
            item['quantity'] = new_quantity
            print(f"Updated quantity of '{name}' to {new_quantity}.")
            return True
    
    print(f"Item '{name}' not found in inventory.")
    return False


def remove_item(inventory, name):
    """
    Remove an item from the inventory.
    
    Args:
        inventory (list): The master inventory list
        name (str): Name of the item to remove
    
    Returns:
        bool: True if removed successfully, False if item not found
    """
    for i, item in enumerate(inventory):
        if item['name'].lower() == name.lower():
            removed_item = inventory.pop(i)
            print(f"Removed '{removed_item['name']}' from inventory.")
            return True
    
    print(f"Item '{name}' not found in inventory.")
    return False


def main():
    """Main function to demonstrate the inventory system."""
    # Initialize empty inventory
    inventory = []
    
    # Demo: Add items
    add_item(inventory, "Laptop", 5, "High-performance gaming laptops")
    add_item(inventory, "Mouse", 15, "Wireless optical mouse")
    add_item(inventory, "Keyboard", 10, "Mechanical keyboards with RGB")
    
    # Display inventory
    display_inventory(inventory)
    
    # Update quantity
    update_quantity(inventory, "Mouse", 20)
    
    # Remove an item
    remove_item(inventory, "Laptop")
    
    # Display updated inventory
    print("\nAfter updates:")
    display_inventory(inventory)


if __name__ == "__main__":
    main()