 # Сумму элементов List

def f01 () : 
    foo = [2,3,4,5,40]
    s = 0
    i = 0
    while i < len(foo) :
        s = s + foo[i]
        i = i + 1
    print (s)

# f01()

# считаем сумму с циклом for in

def f02 () : 
    foo = [2,3,4,5,40]
    s = 0
    for item in foo :
        s = s + item
    print (s)

# f02()

# считаем произведение в list

def f03 () : 
    foo = [2,3,4]
    p = 1
    for item in foo :
        p = p * item
    print (p)

# f03()

# Найти минимальный элемент в List

def f04 () : 
    foo = [2, -3, 4, 22, -11, 22, 56]
    min = foo[0]

    for item in foo :
       if (item < min):
           min = item

    print (min)

# f04()

# Найти максимальный элемент в List

def f05 () : 
    foo = [2, -3, 555, 4, 22, -11, 22, 56]
    max = foo[0]

    for item in foo :
       if (item > max):
           max = item

    print (max)

# f05()

# Найти индекс минимального элемента в List

def f06 () : 
    foo = [2, -3, 444, 22, -11, 22, 56]
    index = 0

    for i in range(len(foo)) :
       if (foo[i] < foo[index]):
           index = i

    print (index)

f06()