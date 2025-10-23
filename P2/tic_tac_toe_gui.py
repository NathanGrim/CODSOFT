"""
Tic-Tac-Toe AI Game with GUI
Beautiful graphical interface using Tkinter
"""

import tkinter as tk
from tkinter import messagebox, font
from tic_tac_toe import TicTacToe
import random


class TicTacToeGUI:
    """GUI class for Tic-Tac-Toe game"""
    
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Tic-Tac-Toe AI")
        self.root.resizable(False, False)
        
        # Color scheme
        self.bg_color = "#2C3E50"
        self.button_bg = "#34495E"
        self.button_hover = "#4A5F7F"
        self.x_color = "#E74C3C"
        self.o_color = "#3498DB"
        self.text_color = "#ECF0F1"
        
        # Game instance
        self.game = TicTacToe()
        self.buttons = []
        self.game_active = True
        self.human_starts = True
        
        # Configure root window
        self.root.configure(bg=self.bg_color)
        
        # Create GUI elements
        self.create_widgets()
        
        # Show start dialog
        self.show_start_dialog()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Title frame
        title_frame = tk.Frame(self.root, bg=self.bg_color)
        title_frame.pack(pady=20)
        
        title_font = font.Font(family="Arial", size=24, weight="bold")
        title_label = tk.Label(
            title_frame,
            text="Tic-Tac-Toe AI",
            font=title_font,
            bg=self.bg_color,
            fg=self.text_color
        )
        title_label.pack()
        
        subtitle_font = font.Font(family="Arial", size=10)
        subtitle_label = tk.Label(
            title_frame,
            text="Powered by Minimax with Alpha-Beta Pruning",
            font=subtitle_font,
            bg=self.bg_color,
            fg="#95A5A6"
        )
        subtitle_label.pack()
        
        # Info frame
        info_frame = tk.Frame(self.root, bg=self.bg_color)
        info_frame.pack(pady=10)
        
        info_font = font.Font(family="Arial", size=12)
        self.info_label = tk.Label(
            info_frame,
            text="Your turn! (X)",
            font=info_font,
            bg=self.bg_color,
            fg=self.text_color
        )
        self.info_label.pack()
        
        # Game board frame
        board_frame = tk.Frame(self.root, bg=self.bg_color)
        board_frame.pack(pady=10)
        
        button_font = font.Font(family="Arial", size=32, weight="bold")
        
        # Create 3x3 grid of buttons
        for i in range(9):
            row = i // 3
            col = i % 3
            
            button = tk.Button(
                board_frame,
                text="",
                font=button_font,
                width=5,
                height=2,
                bg=self.button_bg,
                fg=self.text_color,
                activebackground=self.button_hover,
                relief=tk.RAISED,
                bd=3,
                command=lambda pos=i: self.on_button_click(pos)
            )
            button.grid(row=row, column=col, padx=5, pady=5)
            self.buttons.append(button)
        
        # Control frame
        control_frame = tk.Frame(self.root, bg=self.bg_color)
        control_frame.pack(pady=20)
        
        control_font = font.Font(family="Arial", size=11, weight="bold")
        
        reset_button = tk.Button(
            control_frame,
            text="New Game",
            font=control_font,
            bg="#27AE60",
            fg="white",
            activebackground="#229954",
            command=self.reset_game,
            padx=20,
            pady=10,
            relief=tk.RAISED,
            bd=2
        )
        reset_button.pack(side=tk.LEFT, padx=5)
        
        quit_button = tk.Button(
            control_frame,
            text="Quit",
            font=control_font,
            bg="#C0392B",
            fg="white",
            activebackground="#A93226",
            command=self.root.quit,
            padx=20,
            pady=10,
            relief=tk.RAISED,
            bd=2
        )
        quit_button.pack(side=tk.LEFT, padx=5)
        
        # Stats frame
        stats_frame = tk.Frame(self.root, bg=self.bg_color)
        stats_frame.pack(pady=10)
        
        stats_font = font.Font(family="Arial", size=9)
        stats_label = tk.Label(
            stats_frame,
            text="You: X (Red) | AI: O (Blue)",
            font=stats_font,
            bg=self.bg_color,
            fg="#95A5A6"
        )
        stats_label.pack()
    
    def show_start_dialog(self):
        """Show dialog to choose who goes first"""
        response = messagebox.askyesno(
            "Start Game",
            "Would you like to go first?\n\nYes = You start (X)\nNo = AI starts (O)"
        )
        self.human_starts = response
        
        if not self.human_starts:
            self.info_label.config(text="AI's turn! (O)")
            self.root.after(500, self.ai_move)
        else:
            self.info_label.config(text="Your turn! (X)")
    
    def on_button_click(self, position):
        """Handle button click"""
        if not self.game_active:
            return
        
        if self.game.board[position] != ' ':
            return
        
        # Human move
        self.game.make_move(position, self.game.human)
        self.update_button(position, self.game.human)
        
        # Check if game is over
        if self.check_game_over():
            return
        
        # AI move
        self.info_label.config(text="AI is thinking...")
        self.game_active = False
        self.root.after(500, self.ai_move)
    
    def ai_move(self):
        """Execute AI move"""
        if not self.game_active and self.game.get_available_moves():
            self.game_active = True
            
        move = self.game.get_best_move()
        if move is not None:
            self.game.make_move(move, self.game.ai)
            self.update_button(move, self.game.ai)
            
            if not self.check_game_over():
                self.info_label.config(text="Your turn! (X)")
    
    def update_button(self, position, player):
        """Update button appearance"""
        self.buttons[position].config(
            text=player,
            fg=self.x_color if player == 'X' else self.o_color,
            state=tk.DISABLED,
            disabledforeground=self.x_color if player == 'X' else self.o_color
        )
    
    def check_game_over(self):
        """Check if game is over and handle end game"""
        is_over, winner = self.game.is_game_over()
        
        if is_over:
            self.game_active = False
            
            if winner == self.game.human:
                message = "🎉 Congratulations! You won!\n\n(This is very rare!)"
                self.info_label.config(text="You won! 🎉")
            elif winner == self.game.ai:
                message = "🤖 AI wins!\n\nBetter luck next time!"
                self.info_label.config(text="AI wins! 🤖")
            else:
                message = "🤝 It's a draw!\n\nWell played!"
                self.info_label.config(text="Draw! 🤝")
            
            self.highlight_winner(winner)
            self.root.after(100, lambda: messagebox.showinfo("Game Over", message))
            return True
        
        return False
    
    def highlight_winner(self, winner):
        """Highlight winning combination"""
        if winner in [self.game.human, self.game.ai]:
            win_combinations = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
                [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
                [0, 4, 8], [2, 4, 6]              # Diagonals
            ]
            
            for combo in win_combinations:
                if all(self.game.board[i] == winner for i in combo):
                    for i in combo:
                        self.buttons[i].config(bg="#F39C12")
                    break
    
    def reset_game(self):
        """Reset the game"""
        self.game.reset()
        self.game_active = True
        
        # Reset all buttons
        for button in self.buttons:
            button.config(
                text="",
                state=tk.NORMAL,
                bg=self.button_bg
            )
        
        # Ask who goes first
        self.show_start_dialog()


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = TicTacToeGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()

