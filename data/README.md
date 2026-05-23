# Data Directory

This directory is reserved for generated datasets and checkpoints from later work packages.

## Planned Layout

```text
data/
├── selfplay/
└── checkpoints/
```

## Current Status

No datasets or checkpoints are implemented yet.

## Future Policy

Self-play data and checkpoints should use versioned schemas, including:

- schema version
- engine version or git commit when available
- action-space version
- model checkpoint id
- generation/training settings

Large generated artifacts should generally not be committed unless explicitly needed as small fixtures.
