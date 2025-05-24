# itgid.info - python 2023

# Напишите функцию f04, которая принимает аргумент - list, и возвращает cумму
#  четных элементов данного list. Решите с помощью цикла.

# write your code under this line

def f04 (f) :
    i = 0
    count = 0
    while i < len(f):
        if f[i] % 2 == 0:
            count += f[i]
        i += 1
    return count



b = [1, -2, 4, 5, 2, 7, -11]
result = f04(b)
print (result)
