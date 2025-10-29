# itgid.info - python 2023

# Напишите функцию f15, которая принимает аргумент - list, и возвращает новый list
#  аналогичный полученному но с обратным порядком элементов. Решите с помощью цикла.

# write your code under this line

def f15 (f) :
    reversed_list = []
    i = len(f) - 1
    while i >= 0:
        reversed_list.append(f[i])
        i -= 1
    return reversed_list
        


b = [22, 33, 44, -55, 66]
result = f15(b)
print (result) # одижаю [66, -55, 44, 33, 22]


def f115(f):
    reversed_list = []
    for i in range(len(f) -1, -1, -1):
        reversed_list.append(f[i])
    return reversed_list

b = [22, 33, 44, -55, 66]
result115 = f115(b)
print(result115)