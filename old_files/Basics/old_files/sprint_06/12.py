# itgid.info - python 2023

# Напишите функцию f12, находит и ВОЗВРАЩАЕТ произведение целых чисел от 1 до n. 
# Где n - аргумент функции. 

# например f12(4) функция возвращает число 24, потому что в диапазоне числа 1*2*3*4


# write your code under this line

def f12 (a) :
    total = 1
    i = 1
    while i <= a:
        total *= i
        i += 1
    return total

result = f12(4)
print (result)

