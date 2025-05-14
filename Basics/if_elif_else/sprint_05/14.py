# Task 14
# Напишите функцию f14, которая принимает число и возвращает 'even' если оно четное. Либо 'odd' если нет.

def f14(n):
    
    if n % 2 == 0:
        return 'even'
    else:
        return 'odd'



print(f14(4))