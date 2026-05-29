import random
import pickle
import os

# =============================================================
#  Tic-Tac-Toe with a Q-Learning AI Agent
# =============================================================
#
#  The AI plays as 'O' and learns through self-play during a
#  training phase before you face it. It uses Q-learning to
#  update a table of (state, action) values so that over time
#  it prefers moves that lead to wins or draws.
#
#  Key concepts
#  ------------
#  Q-table  : dict mapping (board_state, action) -> value
#  Alpha    : learning rate  -- how fast new info overwrites old
#  Gamma    : discount factor -- how much future rewards matter
#  Epsilon  : exploration rate -- probability of a random move
# =============================================================

Q_TABLE_FILE = "q_table.pkl"

# ── Hyper-parameters ──────────────────────────────────────────
ALPHA   = 0.5    # learning rate
GAMMA   = 0.9    # discount factor
EPSILON = 0.2    # exploration rate during training
TRAINING_EPISODES = 50_000


# ─────────────────────────────────────────────────────────────
#  1. GAME BOARD
# ─────────────────────────────────────────────────────────────

def create_board():
    """Return an empty 3x3 board as a list of 9 strings."""
    return [" "] * 9


def display_board(board):
    """Print the board with row/column guides."""
    print("\n     1   2   3")
    print("   +---+---+---+")
    for row in range(3):
        cells = board[row * 3: row * 3 + 3]
        print(f" {row + 1} | {cells[0]} | {cells[1]} | {cells[2]} |")
        print("   +---+---+---+")
    print()


def get_available_moves(board):
    """Return indices of empty cells (0-8)."""
    return [i for i, cell in enumerate(board) if cell == " "]


# ─────────────────────────────────────────────────────────────
#  2. WIN / TIE DETECTION
# ─────────────────────────────────────────────────────────────

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def check_winner(board, player):
    """Return True if 'player' has three in a row."""
    return any(
        board[a] == board[b] == board[c] == player
        for a, b, c in WIN_LINES
    )


def check_tie(board):
    """Return True if the board is full with no winner."""
    return " " not in board


# ─────────────────────────────────────────────────────────────
#  3. Q-LEARNING AGENT
# ─────────────────────────────────────────────────────────────

class QLearningAgent:
    """
    Learns to play Tic-Tac-Toe via Q-learning.

    The state is a tuple snapshot of the board.
    Actions are board indices (0-8).
    """

    def __init__(self, player_mark="O"):
        self.player  = player_mark
        self.q_table = {}           # {(state, action): float}
        self.alpha   = ALPHA
        self.gamma   = GAMMA
        self.epsilon = EPSILON

    # ── Q-value helpers ───────────────────────────────────────

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def set_q(self, state, action, value):
        self.q_table[(state, action)] = value

    def best_action(self, state, moves):
        """Return the move with the highest Q-value."""
        return max(moves, key=lambda a: self.get_q(state, a))

    # ── Policy ────────────────────────────────────────────────

    def choose_action(self, board, training=True):
        """
        Epsilon-greedy during training; pure greedy when playing.
        """
        moves = get_available_moves(board)
        state = tuple(board)
        if training and random.random() < self.epsilon:
            return random.choice(moves)           # explore
        return self.best_action(state, moves)     # exploit

    # ── Learning update ───────────────────────────────────────

    def update(self, state, action, reward, next_board):
        next_moves = get_available_moves(next_board)
        next_state = tuple(next_board)

        if next_moves:
            best_next = max(self.get_q(next_state, a) for a in next_moves)
        else:
            best_next = 0.0

        old_q  = self.get_q(state, action)
        new_q  = old_q + self.alpha * (reward + self.gamma * best_next - old_q)
        self.set_q(state, action, new_q)

    # ── Persistence ───────────────────────────────────────────

    def save(self, path=Q_TABLE_FILE):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"  Q-table saved ({len(self.q_table)} entries).")

    def load(self, path=Q_TABLE_FILE):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"  Loaded existing Q-table ({len(self.q_table)} entries).")
            return True
        return False


# ─────────────────────────────────────────────────────────────
#  4. TRAINING  (agent self-play)
# ─────────────────────────────────────────────────────────────

def train_agent(agent, episodes=TRAINING_EPISODES):
    """
    Run self-play games so the agent can populate its Q-table.
    One agent plays both X and O, alternating marks each turn.
    """
    wins = ties = losses = 0

    for _ in range(episodes):
        board      = create_board()
        marks      = ["X", "O"]
        history    = []          # list of (state, action, mark)

        current = 0              # index into marks[]

        while True:
            mark   = marks[current]
            state  = tuple(board)
            action = agent.choose_action(board, training=True)
            board[action] = mark
            history.append((state, action, mark))

            if check_winner(board, mark):
                # Reward the winner, penalise the loser
                for s, a, m in history:
                    reward = 1.0 if m == mark else -1.0
                    agent.update(s, a, reward, board)
                if mark == "O":
                    wins += 1
                else:
                    losses += 1
                break

            if check_tie(board):
                for s, a, _ in history:
                    agent.update(s, a, 0.5, board)
                ties += 1
                break

            current = 1 - current   # switch player

    print(f"  Training complete -- "
          f"O wins: {wins} | Ties: {ties} | X wins: {losses}")


# ─────────────────────────────────────────────────────────────
#  5. MAIN GAME LOOP
# ─────────────────────────────────────────────────────────────

def get_player_move(board):
    """Ask the human for a valid move. Input: 'row col' (1-indexed)."""
    moves = get_available_moves(board)
    while True:
        try:
            raw = input("  Your move (row col, e.g. 1 3): ").strip()
            parts = raw.split()
            if len(parts) != 2:
                raise ValueError
            row, col = int(parts[0]) - 1, int(parts[1]) - 1
            index = row * 3 + col
            if index not in moves:
                print("  That cell is taken or out of range. Try again.")
                continue
            return index
        except (ValueError, IndexError):
            print("  Please enter row and column as two numbers, e.g. '2 3'.")


def play_game(agent):
    """One full game: Human (X) vs AI agent (O)."""
    board = create_board()
    print("\n  You are X. The AI is O.")
    print("  Enter moves as 'row col' (both 1-3).")
    display_board(board)

    current = "X"   # X always goes first

    while True:
        if current == "X":
            # ── Human turn ────────────────────────────────────
            index = get_player_move(board)
        else:
            # ── AI turn ───────────────────────────────────────
            index = agent.choose_action(board, training=False)
            row, col = divmod(index, 3)
            print(f"  AI plays at row {row + 1}, col {col + 1}.")

        board[index] = current
        display_board(board)

        if check_winner(board, current):
            if current == "X":
                print("  Congratulations -- YOU WIN!")
            else:
                print("  The AI wins! Better luck next time.")
            return current          # return winner mark

        if check_tie(board):
            print("  It's a tie! Well played.")
            return "tie"

        current = "O" if current == "X" else "X"   # switch


def main():
    """Main entry point: train (or load) the agent, then play."""
    print("\n" + "=" * 50)
    print("       TIC-TAC-TOE  --  Q-Learning AI")
    print("=" * 50)

    agent = QLearningAgent(player_mark="O")

    # ── Load or train ─────────────────────────────────────────
    if not agent.load():
        print(f"\n  No saved Q-table found.")
        print(f"  Training AI for {TRAINING_EPISODES:,} episodes ...")
        train_agent(agent)
        save_choice = input("  Save Q-table for future sessions? (y/n): ").strip().lower()
        if save_choice in ("y", "yes"):
            agent.save()
    else:
        retrain = input("  Re-train the AI from scratch? (y/n): ").strip().lower()
        if retrain in ("y", "yes"):
            agent.q_table = {}
            print(f"  Training AI for {TRAINING_EPISODES:,} episodes ...")
            train_agent(agent)
            agent.save()

    # ── Game loop ─────────────────────────────────────────────
    scores = {"X": 0, "O": 0, "tie": 0}

    while True:
        result = play_game(agent)
        scores[result] += 1

        print(f"\n  Score -- You: {scores['X']}  |  AI: {scores['O']}  |  Ties: {scores['tie']}")

        again = input("\n  Play again? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("\n  Thanks for playing!\n")
            break


# ─────────────────────────────────────────────────────────────
#  6. CALL THE MAIN GAME LOOP
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
