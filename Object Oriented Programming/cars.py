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
        print(self.gas)

def random_thing():
    print('look at me I am here')
    