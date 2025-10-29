# itgid.info - python 2023

# Напишите функцию f16, которая получает list и формирует строку где значения list
#  идут через подчеркивание. Строка должна быть возвращена.

# write your code under this line
def f16 (ct) :
    out = ''
    i = 0
    while i < len(ct):
        out += str(ct[i]) + '_'
        i += 1
    return out

list1 = [11, 12, 13]
result = f16(list1)
print(result) # ожидаю 11_12_13_