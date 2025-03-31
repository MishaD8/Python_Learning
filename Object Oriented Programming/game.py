from character_class import Character

#Create some characters
hero = Character("Elara", "Mage", 2)
villain = Character("Morkoth", "Necromancer", 4)

print("Adventure begins!")
print(f"Hero: {hero}")
print(f"Villain: {villain}")

# Simulate a battle
print("\n--- Battle starts ---")
damage_to_villain = 25
print(f"{hero.name} casts a fireball for {damage_to_villain} damage!")
villain.take_damage(damage_to_villain)
print(f"Villain status: {villain}")

damage_to_hero = 12
print(f"\n{villain.name} attacks with dark magic for {damage_to_hero} damage!")
hero.take_damage(damage_to_hero)
print(f"Hero status: {hero}")

#Hero heals
heal_amount = 5
print(f"\n{hero.name} drinks a potion and recovers {heal_amount} health")
hero.heal(heal_amount)
print(f"Hero status: {hero}")

# Hero gains experience
print(f"\n{hero.name} gains 200 experience points!")
hero.gain_experience(200)
print(f"Hero status: {hero}")