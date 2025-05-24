# itgid.info - python 2023

# Напишите функцию f10, которая принимает аргумент - list, и возвращает индекс первого
#  встреченного значения list меньше нуля. Решите с помощью цикла.

# write your code under this line

def f10 (f) :
    i = 0
    
    while i < len(f):
        if f[i] < 0:
            return i
        i += 1
    return "no negative numbers"


b = [8, 3, 5,-11, 2, 1, 15, 7]
result = f10(b)
print (result)
