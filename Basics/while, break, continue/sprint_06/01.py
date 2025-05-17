# itgid.info - python 2023

# Выведите в консоль с помощью цикла while числа от 0 до 10 (оба числа включительно). Шаг - единица.

# Ожидается такой вывод в консоли
#   0
#   1
#   2
#   3
#   4
#   5
#   6
#   7
#   8
#   9
#   10

# write your code under this line

def f01(n):
    out = ''
    i = 0
    while i <= n:
        out = out + str(i) + '\n'
        i = i + 1
    print(out)

f01(10)

# option 2

def f02(n):
    i = 0
    while i <= 10:
        print(i)
        i += 1

f02(10)