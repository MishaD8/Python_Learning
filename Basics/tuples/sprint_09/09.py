# itgid.info

# Task 09
# Напишите функцию f09 которая получает два tuple и возвращает их объединение.
#  Причем присоединяет меньший tuple к большему. 

def f09(t1, t2) :
    if len(t1) >= len(t2):
        return t1 + t2
    else:
        return t2 + t1


tpl_1 = (100, 105, 110)
tpl_2 = (555, 666, 777, 33)

res = f09(tpl_1, tpl_2)

print(res)