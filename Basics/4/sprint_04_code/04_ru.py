a = 4
b = 6
c = (a + b) * 10
print (c)

a = 5
b = 3
c = (a + b) * 10
print (c) 



def someFunction () :
    c = (a + b) * 10
    print (c)

a = 1
b = 3
someFunction()
a = 2
b = 4
someFunction()
a = 3
b = 5
someFunction()

# глобальные значения

f = 66

def ex1 () :
    f = 55
    f = f + 22
    print (f)

ex1()
print (f)

print ('\n')

k = 99

def ex2 () :
    # global k
    k = 88
    print (k)

ex2()
print (k)


print ('\n')
# Аргументы

def ex3 (a, b) :
    c = (a + b) * 10
    print (c) 
    return c

ex3(1, 2)
ex3(3, 4)
ex3(100, 50)

m = 100000 + ex3(3,4)
print (m)

def ex4(s) :
    return ('Hello ' + s)

print (ex4('hi'))
print ('Hello hi')

def ex5 (n) :
    return n**2

# print (ex5(3))
result = ex5(5)
print (result)
