import random

correct = random.randint(1, 20)
has_won = False
guesses = 0
while not has_won:
    if guesses > 5:
        print('Sorry too many guesses')
        has_won = True
    else:
        try:
            guess = int(input('Guess a number between 1 and 20  '))
            guesses += 1
            if guess == correct:
                print(f'Congrats You have guessed the correct number of {guesses} guesses')
                has_won = True
            else:
                print('Sorry guess again below.')
        except:
            print('Your guess MUST be a number')
