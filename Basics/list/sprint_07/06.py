# itgid.info - python 2023

# Напишите функцию, которая получает list как аргумент и возвращает 1,
#  если элемент есть в list, и 0 если нет.

# write your code under this line

m6 = [999, 888, 777, 666, 555, 444, 333, 222, 111]

def f06 (lst, element) :
    if element in lst:
        return 1
    else:
        return 0
    

result = f06(m6, 778)
print(result)