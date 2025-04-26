# Вывод нескольких переменных одной командой

a = True
A = 3.14

print (a, A)

# Создание нескольких переменных одной командой

a, b, c = 11, 22, 33
print(a, b, c)

print ("\n")

# Строки

a = 'hello0'
b = "hello"
print (a, b)
print (len (a)) # длина строки

# Конкатенация

c = a + b
print (c)

print ("\n")

f = 'hello'
d = 'Python'
print (f + ' ' + d)
print (d + ' ' + f)

# Типы даннных

g = 3
k = 'cpiO'

print (k + str(g)) # переводим в строку

l = 8
m = '9'

print (l + int(m)) # переводим в число

n = 9.1

print (n + l)

# Форматирование вывода
n = 'Nobody'
b = 'Baby'
print ('{} puts {} in a corner'.format(b, n))

# Дополнительно

x = 11
y = 4
print (x % y) # 4 + 4 + 3

print (4**.5)

print (abs(-5))