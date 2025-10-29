# как добавить элемент в конец списка

def f01 () :
    foo = [11, 22, 33]
    print (foo)
    foo.append(44)
    foo.append(55)
    print (foo)

# f01()

# расширить список за счет другого списка

def f02 () :
    foo = [11, 22, 33]
    bar = [44, 55, 66]
    print (foo, bar)
    foo.extend(bar)
    print (foo)

# f02()

# удалить значение в списке

def f03 () :
    foo = [11, 22, 33, 44, 55]
    print (foo)
    foo.pop()
    print (foo)
    foo.pop(1)
    print (foo)
    del foo[1]
    print (foo)

# f03()

# очистка списка

def f04 () :
    foo = [11, 22, 33, 44, 55]
    print (foo)
    foo = []
    print (foo) 
    ###########
    foo = [11, 22, 33, 44, 55]
    foo.clear()
    print (foo) 

f04()