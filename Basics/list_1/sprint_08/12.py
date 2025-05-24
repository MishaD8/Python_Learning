# itgid.info - python 2023

# Напишите функцию f12, которая принимает аргумент - list, и возвращает индекс максимального
#  значения из list. Решите с помощью цикла.

# write your code under this line

def f12 (f) :
    i = 1
    max_value = f[0]
    max_index = 0
    while i < len(f):
        if f[i] > max_value:
            max_value = f[i]
            max_index = i
        i += 1
    return max_index



b = [2, 3, 11, 5, 6, 9, 12, -4]
result = f12(b)
print (result)
