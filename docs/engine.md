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

`Board` tracks state needed for legal move generation:

- `squares`
- `side_to_move`
- `castling_rights` as a `frozenset` containing any of `KQkq`
- `en_passant_target` as `Square | None`

`Board.starting_position()` creates the standard chess start position with white to move and all castling rights.

`board_from_ascii()` can create placement-oriented test boards from slash-separated rank rows for concise tests.

## FEN Support

Full six-field FEN parsing and serialization lives in `tinychess.engine.fen` and is exported from `tinychess.engine`:

- `STARTING_FEN` / `STANDARD_STARTING_FEN` / `STARTPOS_FEN`
- `FenPosition(board, halfmove_clock, fullmove_number)`
- `parse_fen(fen)`
- `format_fen(position)`
- `board_from_fen(fen)`
- `board_to_fen(board, halfmove_clock=0, fullmove_number=1)`
- `Board.from_fen(fen)` / `board.to_fen(...)`
- `Game.from_fen(fen)` / `game.to_fen()`

`Board` stores placement, side to move, castling rights, and en-passant target. `FenPosition` and `Game` carry the halfmove clock and fullmove number. Serialization uses canonical castling order (`KQkq`) and `-` for absent castling rights or en-passant targets.

## PGN and SAN Support

Bounded PGN parsing/writing and SAN conversion lives in `tinychess.engine.pgn` and is exported from `tinychess.engine`:

- `PgnGame(tags, moves, result, initial_game)`
- `parse_pgn(text)` / `format_pgn(pgn_game)`
- `game_to_pgn(game, tags=None, result=None)`
- `move_to_san(board, move)` / `parse_san(board, san)`
- `Game.from_pgn(text)` / `game.to_pgn(...)`

Supported PGN scope is intentionally small: one mainline game, standard tag pairs, result tokens, FEN setup with `[SetUp "1"]` plus `[FEN "..."]`, and SAN for normal moves, captures, checks, mates, castling, promotions, and disambiguation. `Game.to_pgn()` automatically emits FEN setup tags for non-standard starting positions. SAN parsing is strict: when a legal move gives check or mate, the `+` or `#` suffix must be present. The parser rejects comments, semicolon comments, numeric annotation glyphs, recursive variations, clock annotations, and SAN annotation suffixes rather than silently ignoring them.

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

## Move Application and Transition Strategy

`Board.apply_move(move)` applies a pseudo-legal move and returns a new immutable board snapshot. It updates:

- piece placement
- side to move
- castling rights
- en passant target
- promotion placement
- en passant capture removal
- castling rook movement

The current transition strategy is copy-on-apply. This is simple, safe, and sufficient for the reference implementation and early MCTS work. A make/unmake backend remains a future optimization candidate if benchmarks show board transitions dominate runtime.

## Game State and Outcomes

`Game` lives in `tinychess.engine.game` and is exported from `tinychess.engine`.

It tracks:

- immutable position history
- move history
- halfmove clock
- fullmove number
- repetition counts copied between game snapshots
- optional forced outcome for ply-capped simulations

Primary APIs:

- `Game.new(board=None)`
- `game.board`
- `game.legal_moves`
- `game.outcome`
- `game.play(move)`
- `simulate_game(selector, game=None, max_plies=512)`
- `random_move_selector(seed=None)`

Outcomes use `Outcome` and `OutcomeReason`:

- `CHECKMATE`
- `STALEMATE`
- `FIFTY_MOVE`
- `REPETITION`
- `INSUFFICIENT_MATERIAL`
- `MAX_PLIES`

Draw semantics are pragmatic for complete-game simulation. Repetition and fifty-move style draws are treated as automatic outcomes for now; strict FIDE claim-vs-automatic distinctions are deferred. `Game.play()` rejects moves once an outcome exists, even if the underlying board would still have legal moves.

## Perft and Benchmarks

Run tests:

```bash
uv run pytest tests/test_legal_moves.py tests/test_game.py tests/test_fen.py tests/test_pgn.py
```

Run the lightweight perft benchmark:

```bash
uv run python scripts/perft.py 3
```

Run a deterministic random complete-game benchmark:

```bash
uv run python scripts/random_game.py --seed 7 --max-plies 40
```

Current known start-position counts covered by tests:

| Depth | Nodes |
| --- | ---: |
| 1 | 20 |
| 2 | 400 |
| 3 | 8902 |

The tests also include a Kiwipete-style castling position and focused special-rule coverage for castling, en passant, promotion, check filtering, game history, checkmate, stalemate, halfmove draws, repetition draws, insufficient material, complete-game simulation, FEN round-trips, and bounded PGN/SAN parsing and writing.
