# write a program that takes an age and will tell you if you are able to drive assuming a legal 
# driving age of 16

age = input('Please enter your age to continue>> ')
age = int(age)

# print(age + 20)


# logic
if age > 16:
    print("here are the car keys")
elif age == 16:
    print("congrats for finally being old enough to drive")
else:
    print('sorry but I cannot let you drive')