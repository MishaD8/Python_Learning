# itgid.info - python 2023

# Напишите функцию f05, которая принимает аргумент - list, и возвращает cумму элементов
#  на нечетных позициях (нечетные индексы). Решите с помощью цикла.

# write your code under this line

def f05 (f) :
    i = 0
    count = 0
    while i < len(f):
        if i % 2 != 0:
            count += f[i]
        i += 1
    return count


b = [1, -2, 4, 5, 2, 7, -11, 22]
result = f05(b)
print (result)
