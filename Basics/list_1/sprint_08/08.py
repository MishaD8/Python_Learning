# itgid.info - python 2023

# Напишите функцию f08, которая принимает аргумент - list, и возвращает
#  максимальное значение из list. Решите с помощью цикла.

# write your code under this line

def f08 (f) :
    i = 1
    max_value = f[0]
    while i < len(f):
        if f[i] > max_value:
            max_value = f[i]
        i += 1
    return max_value


b = [8, 2, -5, 11, 2, 1, 15, 3]
result = f08(b)
print (result)

def f008(f):
    max_value = f[0]
    for item in f:
        if item > max_value:
            max_value = item
    return max_value

b = [8, 2, -5, 11, 2, 1, 15, 3]
result008 = f008(b)
print(result008)