class Character:
    def __init__(self, name, role, level = 1):
        self.name = name
        self.role = role
        self.level = level
        self.health = level * 10
        self.experience = 0

    def gain_experience(self, amount):
        """Add experience and level up if threshold reached"""
        self.experience += amount

        # Check if character should level up (100 XP per 1level)
        if self.experience >= self.level * 100:
            self.level_up()
            return True
        return False
    
    def level_up(self):
        """Increase character level and stats"""
        self.level += 1
        self.health = self.level * 10
        print(f"{self.name} leveled up to level {self.level}!")

    def take_damage(self, amount):
        """Reduce character health"""
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            print(f"{self.name} has been defeated!")
            return False
        return True
    
    def heal(self, amount):
        """Restore character health"""
        max_health = self.level * 10
        self.health = min(self.health + amount, max_health)

    def __str__(self):
        """String representation of charcter"""
        exp = self.experience
        return f"{self.name}, Level {self.level} {self.role} (HP: {self.health}, XP: {exp})"
    
# if run directly
if __name__ == "__main__":
    hero = Character("Aragon", "Warrior", 3)
    print(hero)
    hero.take_damage(15)
    print(hero)
    hero.gain_experience(350)
    print(hero)