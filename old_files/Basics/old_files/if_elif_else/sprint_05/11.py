# Task 11
# Напишите функцию f11 которая проверяет name и password и если он равен 'pupkin' и '007' или 'plushkin' и '999' то возвращает 1 и 0 в противном случае.
#
# напоминаю везде где встречаются символы - это латинские символы.

# def f11(name, password):
#     if name == 'pupkin' and password == '007':
#         return 1
#     if name == 'plushkin' and password == '999':
#         return 2
#     else:
#         return 0

# print(f11('pupkin', '0037'))

def f11(name,password):
    if (name == 'pupkin' and password == '007' or name == 'plushkin' and password == '999'):
        return 1
    else:
        return 0
    
print(f11('pupkin', '0037'))