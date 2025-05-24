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
