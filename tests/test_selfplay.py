import numpy as np
import torch

from engine.board import BLACK, WHITE
from engine.mcts import MCTS
from engine.net import PolicyValueNet
from selfplay import _symmetry, play_game


def test_symmetries_move_the_board_and_the_target_together():
    size = 9
    planes = np.zeros((3, size, size), np.float32)
    planes[0, 1, 2] = 1.0
    planes[2] = 1.0
    policy = np.zeros(size * size, np.float32)
    policy[1 * size + 2] = 1.0
    seen = set()
    for k in range(8):
        turned, target = _symmetry(planes, policy, k)
        assert int(turned[0].argmax()) == int(target.argmax())
        seen.add(int(target.argmax()))
    assert len(seen) == 8


def test_self_play_game_records_every_position():
    torch.manual_seed(0)
    net = PolicyValueNet(channels=8, depth=2)
    net.eval()
    history, winner, length = play_game(MCTS(net, torch.device("cpu")), 9, 8, 4, np.random.default_rng(0))
    assert winner in (0, BLACK, WHITE)
    assert len(history) == length - 1
    for planes, policy, side in history:
        assert planes.shape == (3, 9, 9)
        assert abs(policy.sum() - 1.0) < 1e-5
        assert side in (BLACK, WHITE)
