# Применение цикла for
def f01 () :
    foo = [11, 22, 33, 44, 55]
    out = ''
    for item in foo :
        out += str(item) + '_'
    print (out)

# f01()

# Вывод списка с индексами
def f02 () :
    foo = [11, 22, 33, 44, 55]
    out = ''
    i = 0
    while i < len(foo) :
        out += str(i) + ' - ' + str(foo[i]) + '\n'
        i = i + 1
    print (out)

# f02()

# Вывод списка с индексами - for
def f03 () :
    foo = [11, 22, 33, 44, 55]
    out = ''
    for i in range(len(foo)) :
        out += str(i) + ' -> ' + str(foo[i]) + '\n'
    
    print (out)

f03()