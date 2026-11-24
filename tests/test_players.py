from engine.board import BLACK, WHITE, Board
from engine.players import make_player
from engine.search import best_move


def _midgame() -> Board:
    board = Board()
    for (row, col), side in zip(
        [(7, 7), (7, 8), (8, 8), (6, 6), (8, 6), (9, 9)],
        [BLACK, WHITE, BLACK, WHITE, BLACK, WHITE],
    ):
        board.place(row, col, side)
    return board


def test_fixed_depth_gives_the_same_move_every_time():
    board = _midgame()
    first = best_move(board, BLACK, time_limit=None, max_depth=2)
    second = best_move(board, BLACK, time_limit=None, max_depth=2)
    assert first is not None and second is not None
    assert (first.row, first.col, first.score) == (second.row, second.col, second.score)


def test_specs_build_players_that_move():
    board = _midgame()
    for spec in ("base", "base:depth=1", "random:seed=3"):
        player = make_player(spec)
        try:
            result = player.move(board, BLACK)
            assert result is not None
            assert board.get(result.row, result.col) == 0
        finally:
            player.close()


def test_players_still_take_a_win():
    board = Board()
    for col in range(3, 7):
        board.place(7, col, WHITE)
    board.place(0, 0, BLACK)
    player = make_player("base:depth=1")
    result = player.move(board, WHITE)
    assert result is not None
    assert (result.row, result.col) in {(7, 2), (7, 7)}


def test_unknown_spec_is_rejected():
    try:
        make_player("nonsense")
        raise AssertionError("expected an error")
    except ValueError:
        pass
