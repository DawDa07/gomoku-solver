from engine.board import BLACK, WHITE, Board
from engine.learned import white_move
from engine.tactics import find_tactic


def _center() -> Board:
    board = Board()
    board.place(7, 7, BLACK)
    return board


def test_a_white_five_beats_the_book_reply():
    board = _center()
    for col in range(4):
        board.cells[col] = WHITE
    result = white_move(board, WHITE)
    assert result is not None
    assert (result.row, result.col) == (0, 4)
    assert (result.row, result.col) not in {(6, 7), (6, 8)}


def test_an_open_four_beats_the_book_reply():
    board = _center()
    for col in (1, 2, 4):
        board.cells[col] = WHITE
    tactic = find_tactic(board, WHITE)
    assert tactic is not None and len(tactic.moves) == 1
    result = white_move(board, WHITE)
    assert result is not None
    assert (result.row, result.col) == tactic.moves[0]
    assert (result.row, result.col) not in {(6, 7), (6, 8)}


def test_the_center_still_takes_the_longest_delay():
    result = white_move(_center(), WHITE)
    assert result is not None
    assert (result.row, result.col) in {(6, 7), (6, 8)}
    assert "delay" in result.reason.lower()


def test_an_off_book_white_win_is_played():
    board = Board()
    pairs = [(0, 0), (3, 0), (0, 1), (3, 1), (0, 2), (3, 2), (0, 3), (3, 3)]
    for index, (row, col) in enumerate(pairs):
        board.place(row, col, BLACK if index % 2 == 0 else WHITE)
    result = white_move(board, WHITE)
    assert result is not None
    assert (result.row, result.col) == (3, 4)
