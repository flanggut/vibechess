# Engine Documentation

## Board Representation

The current engine uses a compact array/mailbox-style board stored as a 64-entry tuple:

- `None` means an empty square.
- `Piece(color, kind)` means an occupied square.
- Board instances are immutable; mutation-style helpers return new `Board` values.

Square indexing follows the project convention:

```text
a1 == 0
h1 == 7
a2 == 8
a8 == 56
h8 == 63
```

Helpers live in `tinychess.engine.square`:

- `make_square(file_index, rank_index)`
- `parse_square("e4")`
- `square_name(square)`
- `file_index(square)`
- `rank_index(square)`

## Pieces and Moves

Pieces are represented by:

- `Color.WHITE` / `Color.BLACK`
- `PieceType.PAWN`, `KNIGHT`, `BISHOP`, `ROOK`, `QUEEN`, `KING`
- `Piece(color, kind)`

Moves are represented by `Move(from_square, to_square, promotion=None)` and support UCI long algebraic notation:

```python
Move.from_uci("e2e4").to_uci() == "e2e4"
Move.from_uci("e7e8q").promotion == PieceType.QUEEN
```

Move objects do not perform legality checks themselves.

## Board State

`Board` currently tracks the minimum state needed for legal move generation:

- `squares`
- `side_to_move`
- `castling_rights` as a `frozenset` containing any of `KQkq`
- `en_passant_target` as `Square | None`

`Board.starting_position()` creates the standard chess start position with white to move and all castling rights.

`board_from_ascii()` can create placement-oriented test boards from slash-separated rank rows. It is intentionally not full FEN support; full FEN is planned for WP05.

## Legal Move Generation

Legal move generation lives in `tinychess.engine.legal_moves` and is exported from `tinychess.engine`:

- `pseudo_legal_moves(board)`
- `legal_moves(board)`
- `is_in_check(board, color)`
- `is_square_attacked(board, square, by_color)`
- `perft(board, depth)`

Implemented rules include:

- Pawn pushes, double pushes, captures, promotions, and en passant.
- Knight, bishop, rook, queen, and king moves.
- Castling, including occupied path checks, attacked transit/destination checks, and in-check restrictions.
- Filtering of moves that leave the moving side's king in check.

## Minimal Move Application

`Board.apply_move(move)` exists to support legal move filtering and perft. It applies pseudo-legal moves and updates:

- piece placement
- side to move
- castling rights
- en passant target
- promotion placement
- en passant capture removal
- castling rook movement

It intentionally does **not** yet track:

- game history
- halfmove clock
- fullmove number
- repetition
- draw rules
- checkmate/stalemate outcomes

Those belong to WP04.

## Perft

Run tests:

```bash
uv run pytest tests/test_legal_moves.py
```

Run the lightweight perft benchmark:

```bash
uv run python scripts/perft.py 3
```

Current known start-position counts covered by tests:

| Depth | Nodes |
| --- | ---: |
| 1 | 20 |
| 2 | 400 |
| 3 | 8902 |

The tests also include a Kiwipete-style castling position and focused special-rule coverage for castling, en passant, promotion, and check filtering.
