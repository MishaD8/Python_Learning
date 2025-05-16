# i = 0
# while i <= 10 :
#     i = i + 2 # i += 1
#     print (i)
    

def f01 (n) :
    out = ''
    i = 0
    while i <= n :
        out = out + str(i) + '_'
        i = i + 2
    print (out)

# f01(15)

def f02 (n) :
    out = ''
    i = n
    while i >= 0 :
        out += str (i) + '_'
        i = i - 1 # i -= 1
    print (out)

# f02(10)

def f03 (n) :
    out = ''
    i = 0
    while i < n :
        out += '*_'
        i = i + 1
    
    print (out)

# f03(5)

def f04 (n) :
    out = ''
    i = 1999
    while i < n :
        if i % 4 == 0 :
            out += str(i) + '_'
        i = i + 1
    
    print (out)

# f04(2023)

def f05 () :
    i = 5
    out = ''
    while i < 100 :
        if i > 50 :
            break
        out += str(i) + '='
        i += 5 # i = i + 2
    print (out)

# f05()

def f06 () :
    i = 5
    out = ''
    while i < 100 :
        i += 5
        if i == 50 or i == 60 :
            continue
        out += str(i) + '='
        
    print (out)

# f06()

def f07 () :
    s = 1 
    i = 1
    while i <= 4 :
        s = s * i
        i = i + 1
    print (s)

f07()