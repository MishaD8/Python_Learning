# itgid.info - python 2023

# Напишите функцию f11, которая принимает аргумент - list, и возвращает первое значение из
#  list меньше нуля. Решите с помощью цикла.

# write your code under this line

def f11 (f) :
    i = 0
    while i < len(f):
        if f[i] < 0:
            return f[i]
        i += 1
    return "the are no negative numbers"


b = [8, 3, 5, 11, -2, 1, 15, 7]
result = f11(b)
print (result)

def f111(f):
    for i in range (len(f)):
        if f[i] < 0:
            return f[i]
    return "there are no negative numbers" 

b = [8, 3, 5, 11, -2, 1, 15, 7]
result111 = f111(b)
print(result111)
