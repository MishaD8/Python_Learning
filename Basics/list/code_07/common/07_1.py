# Создание списка List

foo_1 = [22, 33, 44, 55] # 0 - 22, 1 - 33, 2 - 44, 3 - 55
foo_2 = [33, 'Hello', True, 3.4]
print (foo_2)

# Обращение к элементу List по индексу
print ('\n')

print (foo_1[0])
print (foo_1[1])
print (foo_1[2])
print (foo_1[3])
# print (foo_1[10]) ошибка!!!!
print (foo_1[-2])

# Обращение к элементу List по диапазону индексов
print ('\n')

foo_1 = [22, 33, 44, 55] # 0 - 22, 1 - 33, 2 - 44, 3 - 55

print (foo_1[1:3]) # 1, 2
print (foo_1[:2]) # 0, 1
print (foo_1[2:]) # 2 и до конца

# Проверка наличия элемента внутри List
print ('\n')

if '55' in foo_1 :
    print ('yes')
else :
    print ('no')

print ('\n')

b = 333
print (b in foo_1)