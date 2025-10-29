# itgid.info - python 2023

# Напишите функцию f19, которая получает list и возвращает сумму элементов в нем. 
# Подсказка - посмотрите sprint по циклам!!!


# write your code under this line
def f19 (ct) :
    i = 0
    total = 0
    while i < len(ct):
        total += ct[i]
        i += 1
    return total


list1 = [5, 6, 7]

s = f19(list1)
print(s) # ожидаю 18