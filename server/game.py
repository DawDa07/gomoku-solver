"""One local game. Play mode answers with the bot; Analyze mode only suggests."""

from __future__ import annotations

from engine.board import BLACK, EMPTY, WHITE, Board, opponent, square_name

MODES = ("play", "analyze")


class Game:
    def __init__(self, time_limit: float = 1.0):
        self.time_limit = time_limit
        self.board = Board()
        self.mode = "play"
        self.human_side = BLACK
        self.side_to_move = BLACK
        self.suggestion: dict | None = None
        self.last_bot_reason = ""
        self.engine = "selfplay"

    def new(self, mode: str = "play", human_side: int = BLACK) -> None:
        self._check_mode(mode)
        self._check_player(human_side)
        self.board = Board()
        self.mode = mode
        self.human_side = human_side
        self.side_to_move = BLACK
        self.suggestion = None
        self.last_bot_reason = ""
        if mode == "play" and human_side == WHITE:
            self._play_bot()

    def set_mode(self, mode: str) -> None:
        self._check_mode(mode)
        self.mode = mode
        self.suggestion = None
        self.last_bot_reason = ""

    def set_side(self, side: int) -> None:
        if self.mode != "analyze":
            raise ValueError("Side to move follows the game in Play mode.")
        self._check_player(side)
        self.side_to_move = side
        self.suggestion = None

    def human_move(self, row: int, col: int, player: int | None = None) -> None:
        self.suggestion = None
        if self.mode == "analyze":
            color = self.side_to_move if player is None else player
            self._check_player(color)
            self._place(row, col, color)
            return
        if self.board.winner or self.board.is_full():
            raise ValueError("The game is over.")
        if self.side_to_move != self.human_side:
            raise ValueError("It is the bot's turn.")
        self._place(row, col, self.human_side)
        self.last_bot_reason = ""
        self.side_to_move = opponent(self.human_side)
        if self.board.winner or self.board.is_full():
            return
        self._play_bot()

    def bot_move(self) -> None:
        if self.mode != "play":
            raise ValueError("The bot only moves in Play mode.")
        if self.board.winner or self.board.is_full():
            raise ValueError("The game is over.")
        if self.side_to_move == self.human_side:
            raise ValueError("It is your turn.")
        self._play_bot()

    def analyze(self) -> None:
        if self.mode != "analyze":
            raise ValueError("Switch to Analyze to ask for a move.")
        if self.board.winner:
            raise ValueError("The game is already over.")
        result = self._choose_move()
        if result is None:
            raise ValueError("No moves left.")
        self.suggestion = {
            "row": result.row,
            "col": result.col,
            "score": result.score,
            "line": [{"row": r, "col": c} for r, c in result.line],
            "reason": result.reason,
        }

    def undo(self) -> None:
        self.suggestion = None
        if not self.board.history:
            return
        if self.mode == "analyze":
            self.board.undo()
            return
        last_player = self.board.history[-1][2]
        self.board.undo()
        if (
            last_player != self.human_side
            and self.board.history
            and self.board.history[-1][2] == self.human_side
        ):
            self.board.undo()
        self.last_bot_reason = ""
        if not self.board.history:
            self.side_to_move = BLACK
        else:
            self.side_to_move = opponent(self.board.history[-1][2])

    def to_dict(self) -> dict:
        last = None
        if self.board.history:
            row, col, player = self.board.history[-1]
            last = {"row": row, "col": col, "player": player}
        return {
            "board": self.board.to_rows(),
            "side_to_move": self.side_to_move,
            "mode": self.mode,
            "human_side": self.human_side,
            "winner": self.board.winner,
            "draw": self.board.is_full() and not self.board.winner,
            "last_move": last,
            "suggestion": self.suggestion,
            "engine": self.engine,
            "status": self.status(),
        }

    def status(self) -> str:
        last = self.board.history[-1] if self.board.history else None
        if self.board.winner:
            text = f"{_name(self.board.winner)} wins."
            if last:
                text += f" Last move {square_name(last[0], last[1])}."
            return text
        if self.board.is_full():
            return "Draw."
        if self.mode == "analyze":
            if self.suggestion:
                return self.suggestion["reason"]
            return f"{_name(self.side_to_move)} to move. Place stones, then ask for the best move."
        if self.side_to_move != self.human_side:
            return f"{_name(self.side_to_move)} to move. The bot has not played yet."
        if self.last_bot_reason and last and last[2] != self.human_side:
            return f"Bot played {square_name(last[0], last[1])}. {self.last_bot_reason} Your turn."
        if not self.board.history:
            return "Your turn. You play black."
        return "Your turn."

    def _play_bot(self) -> None:
        result = self._choose_move()
        if result is None:
            return
        self.board.place(result.row, result.col, self.side_to_move)
        self.last_bot_reason = result.reason
        if not self.board.winner and not self.board.is_full():
            self.side_to_move = opponent(self.side_to_move)

    def _choose_move(self):
        """Black plays the solved line. White plays the delay network. Analyze prefers the book."""
        from engine.book import lookup, prove
        from engine.learned import delay_move

        side = self.side_to_move
        # In Play, White is the trained defender and does not consult the book.
        if self.mode == "play" and self.human_side == BLACK:
            return delay_move(self.board, side)
        found = lookup(self.board, side)
        if found.move is not None:
            return found.move
        # A leaf is still a book position: the distance is stored, the move is proved.
        # When the bot is Black, a position off the tree is proved the same way.
        if self.mode == "play" or found.kind == "leaf":
            proved = prove(self.board, side)
            if proved is not None:
                return proved
        return delay_move(self.board, side)

    def _place(self, row: int, col: int, player: int) -> None:
        if not self.board.in_bounds(row, col):
            raise ValueError("That intersection is off the board.")
        if self.board.winner:
            raise ValueError("The game is over.")
        if self.board.get(row, col) != EMPTY:
            raise ValueError("That intersection is taken.")
        self.board.place(row, col, player)

    @staticmethod
    def _check_mode(mode: str) -> None:
        if mode not in MODES:
            raise ValueError("Mode must be play or analyze.")

    @staticmethod
    def _check_player(player: int) -> None:
        if player not in (BLACK, WHITE):
            raise ValueError("Choose black or white.")


def _name(player: int) -> str:
    return "Black" if player == BLACK else "White"
