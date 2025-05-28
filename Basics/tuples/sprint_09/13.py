# itgid.info

# Task 13
# Напишите функцию f13, которая получает tuple и значение, возвращает число которое
#  показывает сколько раз значение встречается в tuple.

def f13(t, v) :
    count = 0
    for item in t:
        if item == v:
            count += 1
    return count


tpl = (100, 105, 110, 100, 105, 105, 12, 102)

res = f13(tpl, 105)

print(res)