from engine.board import BLACK, WHITE
from engine.book import solver_running
from server.game import Game


def test_white_lets_the_bot_open_at_the_center():
    game = Game(time_limit=0.3)
    game.new("play", WHITE)
    assert game.board.get(7, 7) == BLACK
    assert game.side_to_move == WHITE
    assert game.board.history[-1][2] == BLACK
    assert "book" in game.last_bot_reason.lower()
    assert not solver_running()


def test_analyze_on_the_opening_uses_the_book():
    game = Game()
    game.new("analyze", BLACK)
    game.analyze()
    assert (game.suggestion["row"], game.suggestion["col"]) == (7, 7)
    assert "book" in game.suggestion["reason"].lower()
    game.human_move(7, 7, BLACK)
    game.set_side(WHITE)
    game.analyze()
    assert (game.suggestion["row"], game.suggestion["col"]) in {(6, 7), (6, 8)}
    assert "longest delay" in game.suggestion["reason"].lower()
    assert not solver_running()


def test_play_undo_takes_back_the_whole_exchange():
    game = Game(time_limit=0.4)
    game.human_move(7, 7)
    assert len(game.board.history) == 2
    game.undo()
    assert game.board.history == []
    assert game.side_to_move == BLACK


def test_undo_of_the_opening_does_not_let_you_move_as_white():
    game = Game(time_limit=0.2)
    game.new("play", WHITE)
    game.undo()
    assert game.board.history == []
    try:
        game.human_move(7, 7)
        raise AssertionError("white should not move while it is the bot's turn")
    except ValueError as exc:
        assert "bot" in str(exc).lower()


def test_analyze_suggests_without_playing():
    game = Game(time_limit=0.3)
    game.new("analyze", BLACK)
    for col in range(7, 11):
        game.human_move(7, col, BLACK)
    game.set_side(BLACK)
    game.analyze()
    assert (game.suggestion["row"], game.suggestion["col"]) in {(7, 6), (7, 11)}
    assert len(game.board.history) == 4
    assert "win" in game.status().lower()


def test_analyze_undo_removes_one_stone():
    game = Game()
    game.new("analyze", BLACK)
    game.human_move(3, 3, BLACK)
    game.human_move(3, 4, WHITE)
    game.undo()
    assert game.board.get(3, 3) == BLACK
    assert game.board.get(3, 4) == 0
    assert game.side_to_move == BLACK


def test_switching_mode_keeps_the_stones():
    game = Game()
    game.new("analyze", BLACK)
    game.human_move(4, 4, BLACK)
    game.set_mode("play")
    assert game.board.get(4, 4) == BLACK
    assert game.mode == "play"
    assert game.suggestion is None
