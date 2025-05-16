# как мы создавали list
lst = [33, 44, 55]
print(type(lst))

# создаем кортеж
tpl = (77, 88, 99, 100)

print(type(tpl))

# похожести - вывести по индексу
print(lst[1]);
print(tpl[1]);

# похожести - можно получить len
print( len(lst) );
print( len(tpl) );

# похожести - могут содержать дубляжи

foo = ['hello', 1, 2, 'hello']
bar = ('hi', 33, 44, 'hi')

print(foo)
print(bar)

print('\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n')

# отличие - list можно изменять
lst = [33, 44, 55]
print(type(lst))
lst[0] = 100
print(lst)


# отличие - tuple нельзя изменять
tpl = (111, 222, 333)
print(type(tpl))
# tpl[0] = 100 вызовет ошибку
print(tpl)

