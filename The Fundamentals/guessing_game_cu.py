import random

def number_guessing_game():
    # Generate a random number between 1 and 20
    secret_number = random.randint(1, 20)
    
    # Initialize variables for tracking guesses
    max_guesses = 5
    guesses_taken = 0
    
    print("Welcome to the Number Guessing Game!")
    print(f"Guess a number (1-20). You have {max_guesses} attempts.")
    
    # Game loop
    while guesses_taken < max_guesses:
        # Get user input with exception handling
        try:
            guess = input(f"Guess {guesses_taken + 1}/{max_guesses}: ")
            guess = int(guess)
            
            # Validate the range
            if guess < 1 or guess > 20:
                print("Please enter a number between 1 and 20.")
                continue
                
        except ValueError:
            print("Invalid input! Please enter a valid number.")
            continue
        
        # Increment guess counter
        guesses_taken += 1
        
        # Check the guess
        if guess < secret_number:
            print("Too low!")
        elif guess > secret_number:
            print("Too high!")
        else:
            print(f"Correct! The number was {secret_number}!")
            print(f"You won in {guesses_taken} {'guess' if guesses_taken == 1 else 'guesses'}!")
            return  # End the game when correct
        
        # Let the user know how many guesses they have left
        if guesses_taken < max_guesses:
            remaining = max_guesses - guesses_taken
            print(f"{remaining} {'guess' if remaining == 1 else 'guesses'} left.")
    
    # If we exit the loop without a correct guess
    print(f"Game Over! The number was {secret_number}.")

# Run the game
if __name__ == "__main__":
    number_guessing_game()