from engine.board import BLACK, WHITE, Board
from engine.search import best_move


def _four(board: Board, row: int, start: int, player: int) -> None:
    for col in range(start, start + 4):
        board.place(row, col, player)


def test_takes_a_win_in_one():
    board = Board()
    _four(board, 7, 7, BLACK)
    before = list(board.cells)
    result = best_move(board, BLACK, time_limit=0.4)
    assert result is not None
    assert (result.row, result.col) in {(7, 6), (7, 11)}
    assert "win" in result.reason.lower()
    assert board.cells == before


def test_blocks_a_win_in_one():
    board = Board()
    _four(board, 7, 7, WHITE)
    board.place(0, 0, BLACK)
    result = best_move(board, BLACK, time_limit=0.4)
    assert result is not None
    assert (result.row, result.col) in {(7, 6), (7, 11)}
    assert "block" in result.reason.lower()


def test_takes_own_win_before_blocking():
    board = Board()
    _four(board, 0, 0, WHITE)
    _four(board, 2, 0, BLACK)
    result = best_move(board, BLACK, time_limit=0.4)
    assert result is not None
    assert (result.row, result.col) == (2, 4)


def test_double_threat_beats_a_quiet_move():
    """Playing G8 makes two open fours. Extending the side pair does not."""
    board = Board()
    for col in (3, 4, 5):
        board.place(7, col, BLACK)
    for row in (4, 5, 6):
        board.place(row, 6, BLACK)
    board.place(1, 1, BLACK)
    board.place(1, 2, BLACK)
    board.place(10, 10, WHITE)
    result = best_move(board, BLACK, time_limit=0.5)
    assert result is not None
    assert (result.row, result.col) == (7, 6)
    assert result.line[0] == (7, 6)


def test_empty_board_plays_the_center():
    result = best_move(Board(), BLACK, time_limit=0.5)
    assert result is not None
    assert (result.row, result.col) == (7, 7)


def test_smaller_board_opens_at_its_center():
    board = Board(size=9)
    result = best_move(board, BLACK, time_limit=0.2)
    assert result is not None
    assert (result.row, result.col) == (4, 4)


def test_search_reports_a_line_and_a_score():
    board = Board()
    board.place(7, 7, BLACK)
    board.place(7, 8, WHITE)
    result = best_move(board, BLACK, time_limit=0.8)
    assert result is not None
    assert result.line[0] == (result.row, result.col)
    assert board.get(result.row, result.col) == 0
    assert "Line:" in result.reason or "Score" in result.reason or "win" in result.reason.lower()
