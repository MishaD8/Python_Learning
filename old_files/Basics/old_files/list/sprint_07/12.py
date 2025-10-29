# itgid.info - python 2023

# Напишите функцию f12, которая получает list и значение и добавляет значение в конец list.
#  Возвратите list. 
# write your code under this line

def f12 (ct, item) :
    ct.append(item)
    return ct

m12 = [11, 22]
result = f12(m12, 55)
print(result)