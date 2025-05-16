# Создание list

foo_1 = [22, 33, 44, 22]
foo_2 = [33, 'Hello', True, 4.5]
print (foo_1)
print (foo_2)

# Обращение к элементу по индексу
print ('\n')

print (foo_2[0])
print (foo_2[1])
print (foo_2[2])
print (foo_2[3])
# print (foo_2[4])
print (foo_2[-1])

# Обращение к диапазону элементов списка
print ('\n')
print (foo_2[1:3]) # индекс 1, 2

print ('\n')
print (foo_2[:3])  # индексы 0, 1, 2
print ('\n')
print (foo_2[2:]) # индексы 2 и до конца

# Проверка элемента в списке

if ('Hello' in foo_2):
    print ('yes')

b = 443
print (b in foo_1)