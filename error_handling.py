
# age = int(input('Enter you age to see how old you will be in 30 years>> '))

# print(age + 30)

# error - exception

# if you as the programmer do not handle the exception, the hacker will.

try:
    age = int(input('Enter you age to see how old you will be in 30 years>> '))
    print(age + 30)

except:
    print('Something went wrong please try again...')