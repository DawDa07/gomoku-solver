# Gomoku solver

Gomoku bot

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
