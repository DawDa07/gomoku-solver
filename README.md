# Gomoku solver

Play freestyle Gomoku in the browser. Black follows a solved winning line. White takes a win when it sees one, stalls on the book only while that line is still a proved loss, and otherwise plays the attack network. Analyze uses the book when the position is in it.

The board is 15×15. Five or more stones in a row wins. There are no forbidden moves, and Black plays first.

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app
```

Black's solved line also needs Node on `PATH`. The move book is in `vendor/rv2/`. When a book node stores only a distance, `vendor/rv2/solve.js` runs the threat proof. Without Node, those positions fall through to the network.

Open http://127.0.0.1:8000.

Tests:

```bash
pytest
```

Train the attack network White uses once a position leaves the book. It starts from `models/best.pt`, plays against the base bot and against the solved line with occasional random Black moves, and saves `models/white.pt`. It then reports White's wins against the base bot, and the length of solved Black against this White from the center.

```bash
python tools/train_white.py
```

Restart the server after a retrain so it loads the new weights.

## Play

Choose Black or White. If you take White, the bot plays the solved win: the rv2 move book, then its threat proof, then the attack network if the proof has no win. If you take Black, the bot plays White: a threat move first, then a short search when that search sees a win, then the book's longest delay while the position is still a proved loss, then the attack network.

Black opens. A red dot marks the last stone. Undo takes back your move and the bot's reply. Reset starts over.

## Analyze

Switch to Analyze to paint a position. Pick the color you are placing, then click intersections. Pick who is to move — that is separate from the color you are placing — and press Best move. If that sequence is in the book, the reply is the book's: Black's winning move, or the White reply with the most moves left. Otherwise the attack network answers.

The suggested square is drawn as a faded stone with a green ring. Later squares in the main line are green dots. The panel gives the reason. Undo removes one stone.

## How a move is chosen

Both modes call one function and get back a move, a score, and a reason.

Shapes are read from five-cell windows: every run of five cells along a row, column, or diagonal. Counting one color's stones in a window finds solid and broken shapes alike, so `.XX.X.` is seen as a three, the same as `.XXX.`.

The base bot checks threats first, in this order ([`engine/tactics.py`](engine/tactics.py)):

1. Make five.
2. Block the opponent's four.
3. Make an open four, which has two ways to become five.
4. Win by continuous fours: keep making fours, each with one forced block, until a double four or an open four.
5. Make a double threat that one reply cannot stop, directly or after a four.
6. Block the opponent's open three, broken three, or forced fours. Prefer blocks that do not leave them a double threat next move.
7. Take away a double threat the opponent is about to play.

If a check leaves one move, that move is played. If it leaves several safe blocks, alpha-beta chooses among them. In a quiet position, alpha-beta runs for about a second over empty cells within two intersections of a stone, scored by the windows: more stones of one color in an open window is worth more.

The attack network is a policy and value network in `models/white.pt`, with `models/best.pt` as the fallback. The input is three planes (own stones, opponent stones, and a constant), from the side about to move. Residual convolutions see the whole board. It uses the same threat checks; the network chooses quiet moves, and chooses among safe blocks.

15×15 freestyle Gomoku is a known win for Black. The book in `vendor/rv2/gomoku.json` is one such line, folded into eight symmetries after White's first move. It opens at the center. White's replies are stored with a remaining-move count; `h9` and `i9` (our H7 and I7) are the two first replies that reach 35 plies. A node that stores only a number is finished by the threat proof in `vendor/rv2/solve.js`.

## What comes next

1. **Deeper threat search.** Search chains of threes as well as fours (victory by continuous threats), with a time budget. That is the step that moves toward "does not lose."
2. **FPGA.** The hot loop, in the window scan and in the network, is sliding a window along rows, columns, and diagonals. The window scan reads only the stones on the board, so an accelerator can take a board and return counts without the page knowing.
