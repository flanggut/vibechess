# PGN Dataset Ingestion

Tinychess can convert external PGN game collections into the same policy/value
NPZ dataset layout used by self-play training. This is intended for supervised
policy pretraining before iterative self-play.

## Convert PGN to shards

```bash
uv run python scripts/pgn_ingest.py \
  --input ~/data/chess/lichess_elite_2025-11.pgn \
  --output data/selfplay/pgn-elite \
  --shard-samples 50000
```

By default `--max-games` is `0`, meaning no limit: the script processes the full
input file unless interrupted. Use `--max-games N` for smoke runs. Progress
reporting prints counters to stderr every 100 accepted games by default during
long imports without changing the final stdout summary; pass
`--progress-every-games 0` to disable progress output.

The converter writes:

```text
data/selfplay/pgn-elite/
├── manifest.json
├── shard-00000/
│   ├── samples.npz
│   ├── metadata.json
│   └── games.jsonl
└── shard-00001/
    └── ...
```

Each shard is compatible with `tinychess.nn.self_play.load_self_play_dataset`.
`manifest.json` lists all shard directories for shard-wise training.

## Labels

For each played PGN move, ingestion stores:

- `positions`: the encoded board before the move.
- `legal_masks`: legal actions for that position.
- `mcts_policies`: a one-hot policy target for the played move.
- `outcomes`: the final PGN result from the sample side-to-move perspective.

Soft or weighted policy labels are intentionally deferred.

## Parser tolerance and skipped games

The core PGN parser remains strict. Ingestion sanitizes common public-dataset
features before parsing: brace comments, semicolon comments, recursive
variations, NAGs, and `!`/`?` annotation suffixes. Use `--strict` to disable the
sanitizer and skip records that the strict parser rejects.

Non-standard `SetUp`/`FEN` games are skipped so generated `games.jsonl`
records remain replayable from the normal starting position by the existing
dataset validator. Games with unknown `*` results are also skipped because value
labels use the final PGN result.

## Benchmark ingestion hotspots

Use the dry-run benchmark to see where ingestion time is going before writing
shards:

```bash
uv run python scripts/pgn_ingest_benchmark.py \
  --input lichess_elite_2025-11.pgn \
  --max-records 100
```

The report breaks time down by record streaming, FEN tag screening,
sanitization/parsing, replay legality checks, board encoding, legal-mask
creation, policy allocation, and move application. The `parse_sanitize` phase
includes PGN parser and SAN-resolution time, not only sanitizer regex work. Add
`--format json` for machine-readable output or `--profile-output pgn.prof` to
capture cProfile data for drilling into hot parser functions.

## Train from shards

`scripts/train.py` auto-detects `manifest.json` under `--dataset` and trains one
shard at a time to keep memory bounded:

```bash
uv run python scripts/train.py \
  --dataset data/selfplay/pgn-elite \
  --output data/checkpoints/pgn-pretrain-001 \
  --epochs 1 \
  --batch-size 64
```

The current checkpoint format stores model weights, not optimizer state. During
shard-wise training, model weights and training step continue across shards, but
optimizer state is reinitialized per shard.
