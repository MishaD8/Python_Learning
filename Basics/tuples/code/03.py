# Хотя tuple нельзя менять, их можно объединять

foo = (11, 22, 33)
bar = (44, 55, 66)
# foo += bar
# foo = foo + bar
foo = bar + foo

print (foo)

# удалить turple можно полностью

# del bar
print (bar)