# itgid.info

# Task 10
# Напишите функцию f10 которая перебирает tuple циклом и возвращает в виде строки вида:
# (22, 33, 44) возвращает 22=33=44=
# (55, 1000, 'he') возвращает 55=1000=he=

def f10(t1) :
    result = ""
    for item in t1:
        result += str(item) + "="
    return result


tpl = (100, 105, 110)
tpl1 = (55, 1000, 'he')

res = f10(tpl1)

print(res)