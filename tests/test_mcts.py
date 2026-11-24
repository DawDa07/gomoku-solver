import torch

from engine.board import BLACK, WHITE, Board
from engine.mcts import MCTS, mcts_move
from engine.net import PolicyValueNet


def _search() -> MCTS:
    torch.manual_seed(0)
    net = PolicyValueNet(channels=8, depth=2)
    net.eval()
    return MCTS(net, torch.device("cpu"))


def test_visits_add_up_to_the_playouts():
    board = Board(size=9)
    board.place(4, 4, BLACK)
    board.place(4, 5, WHITE)
    moves, visits, value = _search().search(board, BLACK, 64)
    assert len(moves) == len(visits)
    assert visits.sum() == 64
    assert -1.0 <= value <= 1.0
    assert board.history == [(4, 4, BLACK), (4, 5, WHITE)]


def test_takes_the_win_through_tactics():
    board = Board(size=9)
    for col in range(4):
        board.place(4, col, BLACK)
    for col in range(3):
        board.place(6, col, WHITE)
    board.place(8, 8, WHITE)
    result = mcts_move(_search(), board, BLACK, 32)
    assert (result.row, result.col) == (4, 4)


def test_search_finds_the_open_four_threat():
    """White must block Black's open three; the tactic leaves the blocks to the search."""
    board = Board(size=9)
    for col in (3, 4, 5):
        board.place(4, col, BLACK)
    board.place(0, 0, WHITE)
    board.place(8, 8, WHITE)
    result = mcts_move(_search(), board, WHITE, 64)
    assert (result.row, result.col) in {(4, 2), (4, 6)}


def test_forced_block_inside_the_tree():
    """After any quiet White move, Black's open four wins, so the search sees every quiet move lose."""
    board = Board(size=9)
    for col in (2, 3, 4):
        board.place(4, col, BLACK)
    board.place(0, 0, WHITE)
    board.place(8, 8, WHITE)
    board.place(0, 8, BLACK)
    moves, visits, _ = _search().search(board, WHITE, 400)
    best = int(moves[visits.argmax()])
    assert divmod(best, 9) in {(4, 1), (4, 5)}
