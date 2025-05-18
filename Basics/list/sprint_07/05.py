# itgid.info - python 2023

# Напишите функцию, которая получает list как аргумент и возвращает True, если элемент есть в list,
# и False если нет.

# write your code under this line

m5 = [999, 888, 777, 666, 555, 444, 333, 222, 111]

def f05 (lst, element) :
    return element in lst
    

result = f05(m5, 775)
print(result)