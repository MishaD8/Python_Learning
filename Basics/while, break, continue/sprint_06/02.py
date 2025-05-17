# itgid.info - python 2023

# С помощью цикла while создайте переменную out, где будет лежать строка вида '5_6_7_8_9_'.
#  Выведите в консоль out.  Шаг - единица.

# write your code under this line



def f01(n):
    i = 5
    out = ''
    while i < 10:
        out += str(i) + '_'
        i += 1
    print(out)

#f01(11)

# option 2 without underlines

def f02(n):
    i = 5
    out = ''
    while i < 10:
        out += str(i)
        if i < 9:
            out += '_'
        i += 1
    print(out)

f02(10)
