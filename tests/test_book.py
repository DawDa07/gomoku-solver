from engine.board import BLACK, WHITE, Board
from engine.book import (
    book_key,
    iter_white_targets,
    lookup,
    parse_book_key,
    solver_running,
    stored_replies,
)


def test_center_and_h9_use_the_book_coordinates():
    assert book_key(7, 7) == "h8"
    assert parse_book_key("h8") == (7, 7)
    assert parse_book_key("h9") == (6, 7)


def test_longest_replies_to_the_center_are_the_two_slow_moves():
    board = Board()
    board.place(7, 7, BLACK)
    replies = stored_replies(board)
    assert replies is not None
    best = max(count for _row, _col, count in replies)
    longest = sorted((row, col) for row, col, count in replies if count == best)
    assert best == 34
    assert longest == [(6, 7), (6, 8)]


def test_a_move_outside_the_tree_is_a_miss():
    board = Board()
    board.place(0, 0, BLACK)
    assert lookup(board, BLACK).kind == "miss"
    assert stored_replies(board) is None
    assert not solver_running()


def test_white_targets_start_with_the_long_replies():
    history, targets = next(iter_white_targets())
    assert history == [(7, 7, BLACK)]
    assert set(targets) == {(6, 7), (6, 8)}


def test_a_symmetric_white_move_stays_in_the_book():
    board = Board()
    board.place(7, 7, BLACK)
    board.place(7, 0, WHITE)
    found = lookup(board, BLACK)
    assert found.kind == "move"
    assert found.move is not None
    assert board.get(found.move.row, found.move.col) == 0
