class House:
    def __init__(self, floors, square_meters, windows):
        self.floors = floors
        self.square_meters = square_meters
        self.windows = windows
    
    def __str__(self):
        fl = self.floors
        sq = self.square_meters
        w = self.windows
        return f"House with {fl} floors, {sq} square meters, and {w} windows"
                

# keep testing this file directly
if __name__ == "__main__":
    hut = House(5, 100, 20)
    print(hut)