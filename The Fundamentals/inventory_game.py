inventory_list = []

def add_item():
    try:
        name = input('Name:  ')
        q = int(input('Quantity:  '))
        desc = input('Description:  ')
        new_item = {
            'name': name,
            'q': q,
            'desc': desc
        }
        inventory_list.append(new_item)
    except:
        print('Something went wrong')

def display_items():
    for item in inventory_list:
        name = item['name']
        q = item['q']
        desc = item['desc']
        index = inventory_list.index(item)
        print('----------------------')
        print(f'Item Number: {index}')
        print(f'Item Name: {name}')
        print(f'Item Quantity: {q}')
        print(f'Item Description: {desc}')
        print('----------------------')

def change_quantity():
    try:
        index = int(input('Enter Item Number'  ))
        item = inventory_list[index]
        new_quantity = int(input('Enter New Quantity  '))
        item['q'] = new_quantity
        print(f'The quantity has been changed to: {new_quantity}')
    except:
        print('Error')

def remove_item():
    try:
        index = int(input('Enter Item Number  '))
        # pop() takes in index as an arg and removes it
        inventory_list.pop(index)
    except:
        print('Something went wrong')

add_item()
add_item()
display_items()
remove_item()
display_items()
