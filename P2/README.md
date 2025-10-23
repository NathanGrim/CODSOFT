# Tic-Tac-Toe AI 🎮

An unbeatable Tic-Tac-Toe AI implementation using the **Minimax algorithm with Alpha-Beta Pruning**. This project demonstrates game theory concepts and search algorithms through both console and graphical interfaces.

## 🌟 Features

- **Unbeatable AI**: Implements Minimax algorithm with Alpha-Beta pruning optimization
- **Two Play Modes**: Console-based and beautiful GUI interface
- **Smart Gameplay**: AI makes optimal moves using game theory principles
- **User-Friendly**: Clean, modern interface with helpful prompts
- **Choice of First Player**: You can choose who goes first
- **Visual Feedback**: Color-coded moves and winning combinations

## 🧠 Algorithm Explanation

### Minimax Algorithm
The Minimax algorithm is a decision-making algorithm used in game theory. It works by:
1. Exploring all possible game states
2. Assigning scores to terminal states (win/loss/draw)
3. Choosing the move that maximizes the AI's chances while minimizing the opponent's

### Alpha-Beta Pruning
This optimization technique reduces the number of nodes evaluated in the game tree:
- **Alpha**: Best value for the maximizer (AI)
- **Beta**: Best value for the minimizer (human)
- Prunes branches that won't affect the final decision
- Significantly improves performance without changing the result

## 📋 Requirements

- Python 3.7 or higher
- tkinter (usually comes pre-installed with Python)

## 🚀 Installation

1. Clone or download this repository:
```bash
cd p2
```

2. No external dependencies required! All libraries used are part of Python's standard library.

## 🎯 How to Play

### Console Version
Run the console version for a simple text-based interface:
```bash
python tic_tac_toe.py
```

**Features:**
- Board positions numbered 1-9
- Simple text input
- Clear win/loss/draw messages
- Option to play multiple games

### GUI Version (Recommended)
Run the GUI version for a beautiful graphical interface:
```bash
python tic_tac_toe_gui.py
```

**Features:**
- Modern, colorful interface
- Click buttons to make moves
- Visual highlighting of winning combinations
- Easy reset and quit options
- Status updates for each turn

## 🎮 Game Rules

1. The game is played on a 3×3 grid
2. You are **X** (red), AI is **O** (blue)
3. Players take turns placing their marks
4. First to get 3 marks in a row (horizontal, vertical, or diagonal) wins
5. If all 9 squares are filled without a winner, it's a draw

## 🏆 Can You Beat the AI?

The AI uses an optimal strategy, making it **unbeatable** when playing correctly. The best you can achieve is a **draw**! Try different strategies:
- Start in the center
- Start in a corner
- Try different opening moves

## 📊 Project Structure

```
p2/
├── tic_tac_toe.py       # Core game logic and console interface
├── tic_tac_toe_gui.py   # GUI implementation using tkinter
└── README.md            # This file
```

## 🔍 Code Highlights

### Game Logic (`TicTacToe` class)
- Board representation as a list
- Win detection for all 8 possible combinations
- Move validation and game state management

### AI Implementation
```python
def minimax(self, depth, is_maximizing, alpha, beta):
    # Check terminal states
    if self.is_winner(self.ai):
        return 10 - depth
    elif self.is_winner(self.human):
        return depth - 10
    elif self.is_board_full():
        return 0
    
    # Recursive minimax with alpha-beta pruning
    # ...
```

### GUI Features
- Modern color scheme with dark theme
- Responsive button interactions
- Real-time game status updates
- Smooth AI move transitions with delays

## 🎓 Learning Outcomes

This project helps you understand:
1. **Game Theory**: Zero-sum games and optimal strategies
2. **Search Algorithms**: Tree traversal and state space exploration
3. **Optimization**: Alpha-beta pruning for efficiency
4. **Recursion**: Implementing recursive algorithms
5. **GUI Development**: Creating interactive interfaces with tkinter
6. **Software Design**: Separating logic from presentation

## 🛠️ Technical Details

- **Time Complexity**: O(b^d) where b is branching factor and d is depth
  - With pruning: Much faster in practice
- **Space Complexity**: O(d) for recursion stack
- **Optimizations**:
  - Alpha-Beta pruning reduces search space
  - Depth-based scoring prefers faster wins
  - Random first move for variety

## 🤝 Contributing

Feel free to fork this project and add your own features:
- Different difficulty levels
- Score tracking across multiple games
- Network multiplayer
- Different board sizes (4×4, 5×5)
- Alternative AI algorithms

## 📝 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Created as part of a game AI development project to demonstrate Minimax algorithm implementation.

---

**Enjoy the game and happy coding! 🎉**

