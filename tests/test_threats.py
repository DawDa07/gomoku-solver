import time

from engine.board import BLACK, WHITE, Board
from engine.search import best_move
from engine.threats import fork_moves, open_four_moves, scan, vcf, vct, vct_moves


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


def _chain_position() -> Board:
    """Black has no four and no double threat yet, but threes lead to one."""
    board = Board()
    _stones(board, BLACK, [(7, 6), (7, 7), (5, 8), (6, 10), (6, 11)])
    _stones(board, WHITE, [(0, 0), (14, 14), (0, 14), (14, 0), (2, 2)])
    return board


def test_finds_a_chain_of_threes():
    board = _chain_position()
    before = list(board.cells)
    assert vcf(board, BLACK) is None
    assert fork_moves(board, BLACK) == []
    line = vct(board, BLACK)
    assert line is not None
    assert board.cells[line[0]] == 0
    assert board.cells == before
    assert line[0] in vct_moves(board, BLACK)


def test_tactic_plays_the_chain_of_threes():
    board = _chain_position()
    result = best_move(board, BLACK, time_limit=None, max_depth=1)
    assert result is not None
    assert "threes and fours" in result.reason
    assert result.row * board.size + result.col in vct_moves(board, BLACK)


def test_tactic_plays_the_chain_of_threes():
    board = _chain_position()
    result = best_move(board, BLACK, time_limit=None, max_depth=1)
    assert result is not None
    assert "threes and fours" in result.reason
    assert result.row * board.size + result.col in vct_moves(board, BLACK)


def test_no_chain_from_two_stones():
    board = Board()
    _stones(board, BLACK, [(7, 7), (7, 8)])
    _stones(board, WHITE, [(6, 7), (0, 0)])
    assert vct(board, BLACK) is None


def test_chain_search_stops_at_the_deadline():
    board = _chain_position()
    assert vct(board, WHITE, deadline=time.perf_counter() - 1) is None


def test_scan_leaves_the_board_unchanged():
    board = Board()
    _stones(board, BLACK, [(7, 5), (7, 6), (7, 8)])
    _stones(board, WHITE, [(6, 6), (8, 8)])
    before = list(board.cells)
    best_move(board, WHITE, time_limit=0.3)
    assert board.cells == before
    assert board.winner == 0
