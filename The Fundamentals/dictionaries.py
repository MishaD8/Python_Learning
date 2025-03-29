# dictionaries 

# key - name it whatever we want
# value - whatever we want

# 'key':value

person = {
    'name': 'Lloyd',
    'job': 'mentor',
    'age': 50,
    'is_employed': True,
    'hobbies': ['jogging', 'eating', 'something else']
}

# print(person['is_employed'])

# print(person)
# person['eats_cheese'] = True
# print(person)

# print(person['hobbies'][0])

# hobbies_list = person['hobbies']

# print(hobbies_list[0])

people = [{'name': 'Lloyd'}, {'name': 'Josh'}, {'name': 'Brenda'}]
for person in people:
    print(person['name'])
