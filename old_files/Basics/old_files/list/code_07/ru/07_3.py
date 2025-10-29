# как добавить элемент в конец списка

def f01 () :
    foo = [11, 22, 33]
    out = ''
    for item in foo :
        out += str(item) + '_'
    print (out)

# f01()

# Если нужно индексы - способ простой

def f02 () :
    foo = [44, 55, 66]
    out = ''
    i = 0
    while i < len(foo) :
        out += str (foo[i]) + '_'
        i = i + 1
    print (out)

# f02()

# Если нужно индексы - способ простой

def f03 () :
    foo = [44, 55, 66, 77, 88]
    out = ''
    for i in range(len(foo)) :
        out += str(foo[i]) + '_'
    print(out)

f03()