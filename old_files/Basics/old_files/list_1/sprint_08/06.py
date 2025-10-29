# itgid.info - python 2023

# Напишите функцию f06, которая принимает аргумент - list, и возвращает произведение
#  элементов, значение которых меньше 10. Решите с помощью цикла.

# write your code under this line

def f06 (f) :
    i = 0
    product = 1
    while i < len(f):
        if f[i] < 10:
            product *= f[i]
        i += 1
    return product


b = [1, 2, 5, 11, 2, 1, 15, 3, 15]
result = f06(b)
print (result)


def f006(f):
    product = 1
    for item in f:
        if item < 10:
            product *= item
    return product
    
b = [1, 2, 5, 11, 2, 1, 15, 3, 15]
result006 = f006(b)
print(result006)