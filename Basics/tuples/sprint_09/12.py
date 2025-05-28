# itgid.info

# Task 12
# Напишите функцию f12, которая получает tuple и возвращает сумму его элементов.

def f12(t1) :
    total = 0
    for item in t1:
        total += item
        
    return total


tpl = (100, 105, 110)

res = f12(tpl)

print(res)