import random

# ─────────────────────────────────────────
#  Rock, Paper, Scissors — CLI Game
# ─────────────────────────────────────────

CHOICES = ["rock", "paper", "scissors"]

RULES = {
    "rock":     {"beats": "scissors", "loses_to": "paper"},
    "paper":    {"beats": "rock",     "loses_to": "scissors"},
    "scissors": {"beats": "paper",    "loses_to": "rock"},
}

EMOJI = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}


def display_banner():
    print("\n" + "=" * 40)
    print("   🪨  Rock  •  Paper 📄  •  Scissors ✂️")
    print("=" * 40)


def get_user_choice():
    """Prompt the user until they enter a valid choice."""
    print("\nChoices: rock | paper | scissors  (or 'q' to quit)")
    while True:
        user_input = input("Your choice: ").strip().lower()
        if user_input in ("q", "quit", "exit"):
            return None
        if user_input in CHOICES:
            return user_input
        print(f"  ⚠️  '{user_input}' is not valid. Please type rock, paper, or scissors.")


def get_computer_choice():
    """Return a random choice for the computer."""
    return random.choice(CHOICES)


def determine_winner(user: str, computer: str) -> str:
    """
    Compare the two choices and return one of:
      'user'  – the user wins
      'computer' – the computer wins
      'tie'   – it's a draw
    """
    if user == computer:
        return "tie"
    if RULES[user]["beats"] == computer:
        return "user"
    return "computer"


def display_result(user: str, computer: str, winner: str):
    """Print a formatted round summary."""
    print(f"\n  You chose   : {EMOJI[user]}  {user.capitalize()}")
    print(f"  Computer chose: {EMOJI[computer]}  {computer.capitalize()}")

    if winner == "tie":
        print("\n  🤝  It's a tie!")
    elif winner == "user":
        print(f"\n  🎉  You win! {user.capitalize()} beats {computer}.")
    else:
        print(f"\n  💻  Computer wins! {computer.capitalize()} beats {user}.")


def display_scoreboard(scores: dict, rounds: int):
    """Print the current scoreboard."""
    print("\n" + "-" * 40)
    print(f"  Rounds played : {rounds}")
    print(f"  Your wins     : {scores['user']}")
    print(f"  Computer wins : {scores['computer']}")
    print(f"  Ties          : {scores['tie']}")
    print("-" * 40)


def play_again() -> bool:
    """Ask the player if they want another round."""
    answer = input("\nPlay again? (y/n): ").strip().lower()
    return answer in ("y", "yes")


def main():
    display_banner()

    scores = {"user": 0, "computer": 0, "tie": 0}
    rounds = 0

    while True:
        user_choice = get_user_choice()

        # Player chose to quit
        if user_choice is None:
            print("\n  Thanks for playing! Final scoreboard:")
            display_scoreboard(scores, rounds)
            print("  Goodbye! 👋\n")
            break

        computer_choice = get_computer_choice()
        winner = determine_winner(user_choice, computer_choice)

        rounds += 1
        scores[winner] += 1

        display_result(user_choice, computer_choice, winner)
        display_scoreboard(scores, rounds)

        if not play_again():
            print("\n  Thanks for playing! See you next time 👋\n")
            break


if __name__ == "__main__":
    main()
