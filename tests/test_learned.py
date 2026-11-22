import torch

from engine.board import BLACK, WHITE, Board
from engine.learned import select_action
from engine.net import PolicyValueNet, encode


def test_learned_takes_a_win_even_if_the_net_is_random():
    board = Board()
    for col in range(7, 11):
        board.place(7, col, BLACK)
    row, col, reason, _value = select_action(PolicyValueNet(), board, BLACK, temperature=1.0)
    assert "win" in reason.lower()
    assert (row, col) in {(7, 6), (7, 11)}


def test_learned_blocks_a_win_even_if_the_net_is_random():
    board = Board()
    for col in range(7, 11):
        board.place(7, col, WHITE)
    board.place(0, 0, BLACK)
    row, col, reason, _value = select_action(PolicyValueNet(), board, BLACK, temperature=1.0)
    assert "block" in reason.lower()
    assert (row, col) in {(7, 6), (7, 11)}


def test_learned_blocks_an_open_three_even_if_the_net_is_random():
    torch.manual_seed(0)
    for _ in range(5):
        board = Board()
        for col in (6, 7, 8):
            board.place(7, col, WHITE)
        board.place(0, 0, BLACK)
        board.place(0, 14, BLACK)
        row, col, reason, _value = select_action(PolicyValueNet(), board, BLACK, temperature=1.0)
        assert (row, col) in {(7, 5), (7, 9)}
        assert "open three" in reason.lower()


def test_encoding_is_from_the_side_to_move():
    board = Board(size=9)
    board.place(4, 4, BLACK)
    white = encode(board, WHITE)
    assert white.shape == (3, 9, 9)
    assert white[1, 4, 4] == 1
    assert white[0, 4, 4] == 0
    black = encode(board, BLACK)
    assert black[0, 4, 4] == 1


def test_network_accepts_both_board_sizes():
    net = PolicyValueNet()
    for size in (9, 15):
        board = Board(size=size)
        logits, value = net(torch.from_numpy(encode(board, BLACK)).unsqueeze(0))
        assert logits.shape == (1, size * size)
        assert value.shape == (1,)
