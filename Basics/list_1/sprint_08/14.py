# itgid.info - python 2023

# Напишите функцию f14, которая принимает аргумент - list, и возвращает True если все элементы 
# list - больше нуля и False в противном случае. Решите с помощью цикла.

# write your code under this line

def f14 (f) :
    i = 0
    while i < len(f):
        if f[i] <= 0:
            return False
        i += 1
    return True


b = [22, 33, 44, -55, 66]
result = f14(b)
print (result)


def f114(f):
    for i in range(len(f)):
        if f[i] <= 0:
            return False
    return True

b = [22, 33, 44, -55, 66]
result114 = f114(b)
print(result114)