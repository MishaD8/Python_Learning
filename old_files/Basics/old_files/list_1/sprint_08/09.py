# itgid.info - python 2023

# Напишите функцию f09, которая принимает аргумент - list, и возвращает минимальное значение
#  из list. Решите с помощью цикла.

# write your code under this line

def f09 (f) :
    i = 1
    min_value = f[0]
    while i < len(f):
        if f[i] < min_value:
            min_value = f[i]
        i += 1
    return min_value


b = [8, 3, 5, 11, 2, 1, 15, 7]
result = f09(b)
print (result)


def f009(f):
    min_value = f[0]
    for item in f:
        if item < min_value:
            min_value = item
    return min_value

b = [8, 3, 5, 11, 2, 1, 15, 7]
result009 = f009(b)
print(result009)