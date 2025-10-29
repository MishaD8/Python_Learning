

# names = ['Lloyd']

#names is an INSTANCE of the class list

# print(type(names))

# a Toyota is an INSTANCE of class car

# class Car:
#     color = "blue" #attributes for color make and model
#     make = "Toyota"
#     model = "Prius"

class Car:
    def __init__(self, color, make, model): # self means this instance of the class NOT the entire class!
        self.color = color
        self.make = make
        self.model = model
        #default value for ALL instances of a class
        self.gas = 0

    def get_gas(self): # add gas to the car being referenced
        self.gas += 10

    def check_gas(self): # check how much gas I have
        return self.gas

# instantiate an object of class Car

# car1 = Car()
# car2 = Car()
# car1.color = "black"

# print(car1.color)
# print(car2.color)

car1 = Car('blue', 'hyundai', 'elantra')
car2 = Car('grey', 'lamborghini', 'urus')
print(car1.check_gas())
car1.get_gas()
print(car1.check_gas())
# print(f'Car1 has a color of {car1.color}, and Car2 has a color of {car2.color}')

