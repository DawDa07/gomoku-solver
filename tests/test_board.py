from engine.board import BLACK, EMPTY, SIZE, WHITE, Board


def test_place_and_undo_round_trip():
    board = Board()
    board.place(7, 7, BLACK)
    board.place(7, 8, WHITE)
    assert board.get(7, 7) == BLACK
    assert board.get(7, 8) == WHITE
    assert board.undo() == (7, 8, WHITE)
    assert board.get(7, 8) == EMPTY
    assert board.undo() == (7, 7, BLACK)
    assert board.undo() is None
    assert board.cells == [EMPTY] * (SIZE * SIZE)


def test_rejects_occupied_and_out_of_bounds():
    board = Board()
    board.place(0, 0, BLACK)
    try:
        board.place(0, 0, WHITE)
        raise AssertionError("occupied cell should be rejected")
    except ValueError:
        pass
    try:
        board.place(-1, 0, BLACK)
        raise AssertionError("out of bounds should be rejected")
    except ValueError:
        pass
    assert board.get(0, 0) == BLACK


def test_four_in_a_row_does_not_win():
    board = Board()
    for col in range(4):
        board.place(7, col, BLACK)
    assert board.winner == EMPTY


def test_horizontal_vertical_and_both_diagonals():
    horizontal = Board()
    for col in range(5):
        horizontal.place(7, col, BLACK)
    assert horizontal.winner == BLACK

    vertical = Board()
    for row in range(5):
        vertical.place(row, 3, WHITE)
    assert vertical.winner == WHITE

    main = Board()
    for i in range(5):
        main.place(i, i, BLACK)
    assert main.winner == BLACK

    anti = Board()
    for i in range(5):
        anti.place(i, 4 - i, WHITE)
    assert anti.winner == WHITE


def test_six_in_a_row_wins():
    rows = [[EMPTY] * SIZE for _ in range(SIZE)]
    for col in range(6):
        rows[4][col] = BLACK
    board = Board.from_rows(rows)
    assert board.winner == BLACK


def test_undo_clears_a_win_and_blocks_further_moves():
    board = Board()
    for col in range(5):
        board.place(2, col, BLACK)
    assert board.winner == BLACK
    try:
        board.place(0, 0, WHITE)
        raise AssertionError("no moves after a win")
    except ValueError:
        pass
    board.undo()
    assert board.winner == EMPTY
    board.place(0, 0, WHITE)
    assert board.get(0, 0) == WHITE


def test_empty_board_offers_the_center():
    assert Board().legal_moves() == [(SIZE // 2, SIZE // 2)]


def test_legal_moves_stay_near_stones():
    board = Board()
    board.place(0, 0, BLACK)
    moves = set(board.legal_moves())
    assert (0, 1) in moves
    assert (2, 2) in moves
    assert (0, 3) not in moves
    assert (3, 0) not in moves
    assert all(board.get(r, c) == EMPTY for r, c in moves)


def test_no_moves_after_a_win():
    board = Board()
    for col in range(5):
        board.place(0, col, BLACK)
    assert board.legal_moves() == []
