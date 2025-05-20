# itgid.info - python 2023

# Напишите функцию f17, которая получает list и формирует строку где значения list идут
#  через подчеркивание - порядок значений - обратный. Строка должна быть возвращена.

# write your code under this line
def f17 (ct) :
    out = ''
    i = len(ct) - 1
    while i >= 0:
        out += str(ct[i]) + '_'
        i -= 1
    return out


list1 = [11, 12, 13]

result = f17(list1)
print(result) # ожидаю 13_12_11_