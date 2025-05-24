# itgid.info - python 2023

# Напишите функцию f13, которая принимает аргумент - list, и возвращает True если сумма элементов
#  list больше 100 и False в противном случае.  Решите с помощью цикла.

# write your code under this line

def f13 (f) :
    i = 0
    total = 0
    while i < len(f):
        total += f[i]
        i += 1
    return total > 100
            
        


b = [22, 33, 44, 55]
result = f13(b)
print (result)


def f113(f):
    total = 0
    for i in range(len(f)):
        total += f[i]
        if total > 100:
            return True
    return False

b = [22, 33, 44, 55]
result113 = f113(b)
print(result113)