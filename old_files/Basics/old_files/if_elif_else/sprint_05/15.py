# Task 15
# Напишите функцию f15, которая принимает логин и пароль. Если логин не пустая строка и пароль больше 5 символов возвращает True. В остальных случаях - False.

def f15(login, password):
    if login != '' and len(password) > 5:
        return True
    else:
        return False
    



print(f15('musk', 'ilonchik_tesla')) # ожилаю false