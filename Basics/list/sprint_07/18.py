# itgid.info - python 2023

# Напишите функцию f18, которая получает list и формирует строку индекс дефис значение
#  перенос строки. Строка должна быть возвращена.

# ожидаемый вывод
# 0-11
# 1-12
# 2-13

# write your code under this line
def f18 (ct) :
    out = ''
    i = 0
    while i < len(ct):
        out += f"{i} - {ct[i]}\n"
        i += 1
    return out

list1 = [11, 12, 13]

result = f18(list1)
print(result) 