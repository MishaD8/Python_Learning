a = 4
b = 6
c = (a + b) * 10
print (c)

a = 2
b = 3
c = (a + b) * 10
print (c)

print ('\n')

def someFunction () :
    c = (a + b) * 10
    print (c)

a = 2
b = 3
someFunction()
a = 4
b = 5
someFunction()
a = 5
b = 6
someFunction()

print ('\n')
# глобальні значення

f = 66

def ex1 () :
    # global f 
    f = 44
    print (f)

ex1()
print (f)

print ('\n')

# аргументи

def ex2 (x, second) :
    c = (x + second) * 10
    print (c)

ex2 (2, 3)
ex2 (4, 6)
ex2 (5, 8)

def ex3 (s) :
    c = 'Hello ' + s
    print (c)

ex3 ('Mary')
ex3 ('Henry')

def ex4 (n, m) :
    result = n + m # 5
    return result

f = 100 + ex4(23, 3)
print (f)
print (ex4(100,200))

def ex5(n) :
    return n**2

print (ex5(5))
c = 100 + ex5(4)
print (c)