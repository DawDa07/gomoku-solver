# Gomoku solver

Play freestyle Gomoku in the browser against either bot, or set up a position and ask for the best next move.

The board is 15×15. Five or more stones in a row wins. There are no forbidden moves, and Black plays first.

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app
```

Open http://127.0.0.1:8000.

Tests:

```bash
pytest
```

Retrain the learned bot (a few minutes on a laptop):

```bash
python train.py
```

Restart the server after a retrain so it loads the new weights.

## Play

Choose Black or White, and choose the opponent.

- **Base** is the search bot. It looks about a second ahead.
- **Learned** is a small convolutional network. It trained by playing the base bot, then by playing itself. Immediate wins and blocks still come from the rules, so a one-move tactic is not left to the network.

Black opens. A red dot marks the last stone. Undo takes back your move and the bot's reply. Reset starts over.

## Analyze

Switch to Analyze to paint a position. Pick the color you are placing, then click intersections. Pick who is to move — that is separate from the color you are placing — and press Best move. The opponent switch chooses which bot answers.

The suggested square is drawn as a faded stone with a green ring. For the base bot, later squares in the main line are green dots. The panel gives the reason. Undo removes one stone.

## How a move is chosen

Both modes call one function and get back a move, a score, and a reason.

The base bot takes a win in one move, and blocks an immediate loss, before it searches. Otherwise it runs alpha-beta for about a second, looking only at empty cells within two intersections of a stone. The score comes from patterns on the four directions: open fours, closed fours, open threes, and smaller shapes.

The learned bot is a policy and value network. The input is three planes (own stones, opponent stones, and a constant), from the side about to move. Residual convolutions see the whole board, so weights trained on 9×9 still play on 15×15. Training starts on 9×9 because a full-size game is slow to repeat. In each position the base search names the move to imitate, and the value target is who won the game. The network then plays the base and plays itself, while the search keeps labeling the moves.

15×15 freestyle Gomoku is a known win for Black with perfect play. That proof is a search result, not something this network has learned. The learned bot is there to study that loop: play the current best policy, label the positions, and train again. It is not a perfect player.

## What comes next

1. **Threat search.** A forcing search that keeps giving check solves sharp positions the pattern search still misses. That is the step that moves toward "does not lose."
2. **FPGA.** The hot loop, in the pattern score and in the network, is sliding a window along rows, columns, and diagonals. The pattern score reads only the stones on the board, so an accelerator can take a board and return scores without the page knowing.
