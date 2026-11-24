import pytest

from engine.board import square_name
from engine.players import make_player
from tools.puzzles import QUICK, load_all, solve

PUZZLES = load_all(QUICK)


def test_every_quick_category_has_puzzles():
    assert {puzzle.category for puzzle in PUZZLES} == set(QUICK)


@pytest.mark.parametrize("puzzle", PUZZLES, ids=lambda puzzle: puzzle.name)
def test_base_solves_quick_puzzles(puzzle):
    player = make_player("base")
    ok, move, _ = solve(player, puzzle)
    played = "nothing" if move is None else square_name(*move)
    assert ok, f"{puzzle.name}: played {played}, answers {sorted(square_name(*s) for s in puzzle.answers)}"
