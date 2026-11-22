from engine.board import BLACK, WHITE, Board
from engine.search import best_move
from engine.threats import open_four_moves, scan, vcf


def _stones(board: Board, player: int, squares) -> None:
    for row, col in squares:
        board.place(row, col, player)


def _index(board: Board, row: int, col: int) -> int:
    return row * board.size + col


def test_broken_shapes_are_found():
    board = Board()
    _stones(board, BLACK, [(7, 5), (7, 6), (7, 8)])
    wins, fours = scan(board, BLACK)
    assert wins == set()
    assert _index(board, 7, 7) in fours
    opens = open_four_moves(board, BLACK)
    assert _index(board, 7, 7) in opens


def test_closed_three_is_not_an_open_three():
    board = Board()
    _stones(board, WHITE, [(7, 4)])
    _stones(board, BLACK, [(7, 5), (7, 6), (7, 7)])
    assert open_four_moves(board, BLACK) == set()


def test_blocks_a_solid_open_three():
    board = Board()
    _stones(board, WHITE, [(7, 6), (7, 7), (7, 8)])
    _stones(board, BLACK, [(0, 0), (0, 14)])
    result = best_move(board, BLACK, time_limit=0.5)
    assert result is not None
    assert (result.row, result.col) in {(7, 5), (7, 9)}
    assert "open three" in result.reason.lower()


def test_blocks_a_broken_three():
    board = Board()
    _stones(board, WHITE, [(7, 6), (7, 7), (7, 9)])
    _stones(board, BLACK, [(0, 0), (0, 14)])
    result = best_move(board, BLACK, time_limit=0.5)
    assert result is not None
    assert (result.row, result.col) in {(7, 5), (7, 8), (7, 10)}
    assert "open three" in result.reason.lower()


def test_blocks_a_diagonal_three():
    board = Board()
    _stones(board, WHITE, [(5, 5), (6, 6), (7, 7)])
    _stones(board, BLACK, [(0, 14), (14, 0)])
    result = best_move(board, BLACK, time_limit=0.5)
    assert result is not None
    assert (result.row, result.col) in {(4, 4), (8, 8)}


def test_makes_own_open_four_before_blocking_a_three():
    board = Board()
    _stones(board, BLACK, [(3, 6), (3, 7), (3, 8)])
    _stones(board, WHITE, [(10, 6), (10, 7), (10, 8)])
    result = best_move(board, BLACK, time_limit=0.5)
    assert result is not None
    assert (result.row, result.col) in {(3, 5), (3, 9)}
    assert "open four" in result.reason.lower()


def test_finds_a_four_three():
    """H8 makes a closed four across row 8 and an open three down column H."""
    board = Board()
    _stones(board, WHITE, [(7, 3), (0, 0)])
    _stones(board, BLACK, [(7, 4), (7, 5), (7, 6), (5, 7), (6, 7)])
    line = vcf(board, BLACK)
    assert line is not None
    assert line[0] == _index(board, 7, 7)
    result = best_move(board, BLACK, time_limit=0.5)
    assert result is not None
    assert (result.row, result.col) == (7, 7)
    assert "fours" in result.reason.lower()


def test_scan_leaves_the_board_unchanged():
    board = Board()
    _stones(board, BLACK, [(7, 5), (7, 6), (7, 8)])
    _stones(board, WHITE, [(6, 6), (8, 8)])
    before = list(board.cells)
    best_move(board, WHITE, time_limit=0.3)
    assert board.cells == before
    assert board.winner == 0
