# itgid.info - python 2023

# Напишите функцию f02, которая принимает аргумент - list,
#  и возвращает произведение его элементов. Решите с помощью цикла.

# write your code under this line

def f02 (f) :
    i = 0
    count = 1
    while i < len(f):
        count *= f[i]
        i += 1
    return count


b = [1, 2, 4, 5, 2]
result = f02(b)
print (result)


def f002(f):
    product = 1
    for item in f:
        product *= item
    return product

b = [1,2,4,5,2]
result002 = f002(b)
print (result002)