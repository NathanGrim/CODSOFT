"""
Tic-Tac-Toe AI Game
Implements an unbeatable AI using Minimax algorithm with Alpha-Beta pruning
"""

import math
import random
from typing import List, Tuple, Optional


class TicTacToe:
    """Main game class for Tic-Tac-Toe"""
    
    def __init__(self):
        """Initialize the game board"""
        self.board = [' ' for _ in range(9)]
        self.human = 'X'
        self.ai = 'O'
        
    def print_board(self):
        """Display the current board state"""
        print("\n")
        print(f" {self.board[0]} | {self.board[1]} | {self.board[2]} ")
        print("---|---|---")
        print(f" {self.board[3]} | {self.board[4]} | {self.board[5]} ")
        print("---|---|---")
        print(f" {self.board[6]} | {self.board[7]} | {self.board[8]} ")
        print("\n")
        
    def print_board_with_positions(self):
        """Display the board with position numbers"""
        print("\nBoard positions:")
        print(" 1 | 2 | 3 ")
        print("---|---|---")
        print(" 4 | 5 | 6 ")
        print("---|---|---")
        print(" 7 | 8 | 9 ")
        print("\n")
        
    def get_available_moves(self) -> List[int]:
        """Return list of available positions"""
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def is_winner(self, player: str) -> bool:
        """Check if the given player has won"""
        # All possible winning combinations
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in win_combinations:
            if all(self.board[i] == player for i in combo):
                return True
        return False
    
    def is_board_full(self) -> bool:
        """Check if the board is full"""
        return ' ' not in self.board
    
    def is_game_over(self) -> Tuple[bool, Optional[str]]:
        """
        Check if game is over
        Returns: (is_over, winner)
        winner can be 'X', 'O', 'Draw', or None
        """
        if self.is_winner(self.human):
            return True, self.human
        elif self.is_winner(self.ai):
            return True, self.ai
        elif self.is_board_full():
            return True, 'Draw'
        return False, None
    
    def make_move(self, position: int, player: str) -> bool:
        """
        Make a move on the board
        Returns True if move was successful, False otherwise
        """
        if self.board[position] == ' ':
            self.board[position] = player
            return True
        return False
    
    def undo_move(self, position: int):
        """Undo a move"""
        self.board[position] = ' '
    
    def minimax(self, depth: int, is_maximizing: bool, alpha: float, beta: float) -> int:
        """
        Minimax algorithm with Alpha-Beta pruning
        
        Args:
            depth: Current depth in the game tree
            is_maximizing: True if AI's turn (maximizing), False if human's turn (minimizing)
            alpha: Best value for maximizer
            beta: Best value for minimizer
            
        Returns:
            Score for the current board state
        """
        # Check terminal states
        if self.is_winner(self.ai):
            return 10 - depth  # Prefer faster wins
        elif self.is_winner(self.human):
            return depth - 10  # Prefer slower losses
        elif self.is_board_full():
            return 0  # Draw
        
        if is_maximizing:
            # AI's turn - maximize score
            max_eval = -math.inf
            for move in self.get_available_moves():
                self.board[move] = self.ai
                eval_score = self.minimax(depth + 1, False, alpha, beta)
                self.board[move] = ' '
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta pruning
            return max_eval
        else:
            # Human's turn - minimize score
            min_eval = math.inf
            for move in self.get_available_moves():
                self.board[move] = self.human
                eval_score = self.minimax(depth + 1, True, alpha, beta)
                self.board[move] = ' '
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta pruning
            return min_eval
    
    def get_best_move(self) -> int:
        """
        Get the best move for AI using Minimax algorithm
        
        Returns:
            Position index for the best move
        """
        best_score = -math.inf
        best_move = None
        available_moves = self.get_available_moves()
        
        # If it's the first move and board is empty, choose center or corner randomly
        if len(available_moves) == 9:
            return random.choice([0, 2, 4, 6, 8])
        
        for move in available_moves:
            self.board[move] = self.ai
            score = self.minimax(0, False, -math.inf, math.inf)
            self.board[move] = ' '
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def reset(self):
        """Reset the game board"""
        self.board = [' ' for _ in range(9)]


def play_console_game():
    """Play Tic-Tac-Toe in console mode"""
    print("=" * 50)
    print("Welcome to Tic-Tac-Toe AI!")
    print("=" * 50)
    print("\nYou are 'X' and the AI is 'O'")
    print("The AI uses Minimax with Alpha-Beta Pruning")
    print("Good luck trying to beat it!")
    
    game = TicTacToe()
    game.print_board_with_positions()
    
    # Ask who goes first
    while True:
        first = input("Who should go first? (1 for You, 2 for AI): ").strip()
        if first in ['1', '2']:
            break
        print("Invalid input. Please enter 1 or 2.")
    
    current_player = game.human if first == '1' else game.ai
    
    while True:
        game.print_board()
        
        if current_player == game.human:
            # Human's turn
            print("Your turn (X)")
            while True:
                try:
                    move = input("Enter position (1-9): ").strip()
                    move = int(move) - 1  # Convert to 0-indexed
                    
                    if move < 0 or move > 8:
                        print("Invalid position. Please enter a number between 1 and 9.")
                        continue
                    
                    if game.make_move(move, game.human):
                        break
                    else:
                        print("That position is already taken. Try again.")
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 9.")
                except KeyboardInterrupt:
                    print("\n\nGame interrupted. Thanks for playing!")
                    return
        else:
            # AI's turn
            print("AI's turn (O)")
            print("AI is thinking...")
            move = game.get_best_move()
            game.make_move(move, game.ai)
            print(f"AI chose position {move + 1}")
        
        # Check if game is over
        is_over, winner = game.is_game_over()
        if is_over:
            game.print_board()
            if winner == game.human:
                print("*** Congratulations! You won! (This shouldn't happen if AI is working correctly)")
            elif winner == game.ai:
                print("*** AI wins! Better luck next time!")
            else:
                print("*** It's a draw! Well played!")
            break
        
        # Switch player
        current_player = game.ai if current_player == game.human else game.human
    
    # Ask to play again
    play_again = input("\nWould you like to play again? (y/n): ").strip().lower()
    if play_again == 'y':
        play_console_game()
    else:
        print("\nThanks for playing! Goodbye!")


if __name__ == "__main__":
    play_console_game()

