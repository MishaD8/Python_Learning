# как добавить элемент в конец списка

def f01 () :
    foo = [11, 22, 33]
    print (foo)
    foo.append(44)
    print (foo)

# f01()

# расширить список за счет другого

def f02 () :
    foo = [11, 22, 33]
    bar = [44, 55, 66]
    print (foo, bar)
    foo.extend(bar)
    print (foo)

# f02()

# удалить индекс

def f03 () :
    foo = [11, 22, 33, 44, 55]
    print (foo)
    foo.pop()
    print (foo)
    foo.pop(2)
    print (foo)

# f03()

# удалить индекс

def f04 () :
    foo = [11, 22, 33, 44, 55]
    print (foo)
    del foo[0] # можно удалять весь list
    print (foo)

# f04()

# очистить список

def f05 () :
    foo = [11, 22, 33, 44, 55]
    print (foo)
    foo = []
    print (foo)

    #######

    foo = [11, 22, 33, 44, 55]
    print (foo)
    foo.clear()
    print (foo)

f05()
