"""ANSI terminal progress rendering for the self-play generation script.

This module holds the pure presentation layer: worker/render state snapshots and
the ANSI renderer that draws them. It has no dependency on the generation
pipeline, so the orchestration glue (event plumbing, threading, the reporter)
stays in ``self_play.py`` and feeds these renderers immutable snapshots.
"""

from __future__ import annotations

import atexit
import shutil
import sys
from dataclasses import dataclass, field
from typing import ClassVar, Literal, TextIO

ProgressStatus = Literal["pending", "running", "completed", "saving", "done", "failed"]


@dataclass(frozen=True, slots=True)
class WorkerProgressState:
    """Immutable snapshot of one worker's progress for rendering."""

    worker_id: int
    start_game: int
    total_games: int
    games_completed: int = 0
    samples: int = 0
    plies: int = 0
    status: ProgressStatus = "pending"

    @property
    def processed_games(self) -> int:
        return self.games_completed


@dataclass(frozen=True, slots=True)
class ProgressRenderState:
    """Immutable snapshot of the whole run, passed to the renderer."""

    total_games: int
    workers: tuple[WorkerProgressState, ...]
    status: ProgressStatus
    message: str | None = None
    detail_lines: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class AnsiProgressRenderer:
    enabled: bool
    total_games: int
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    label: str = "self-play"
    unit_label: str = "samples"
    _rendered_lines: int = 0
    _started: bool = False
    _finished: bool = False
    _restore_registered: bool = False

    _BAR_WIDTH: ClassVar[int] = 24
    # Pad row labels so each progress bar begins in the same column after
    # removing per-worker status from the TUI rows.
    _ROW_LABEL_WIDTH: ClassVar[int] = len("total")
    _CLEAR_LINE: ClassVar[str] = "\x1b[2K"
    _CURSOR_HIDE: ClassVar[str] = "\x1b[?25l"
    _CURSOR_SHOW: ClassVar[str] = "\x1b[?25h"

    def render(self, state: ProgressRenderState) -> None:
        self._draw(state, restore_cursor=False)

    def finish(self, state: ProgressRenderState) -> None:
        self._draw(state, restore_cursor=True)
        if self.enabled:
            self._finished = True

    def _draw(self, state: ProgressRenderState, *, restore_cursor: bool) -> None:
        if not self.enabled or self._finished:
            return
        lines = self._format_lines(state)
        width = self._terminal_width()
        if not self._started:
            self.stream.write(self._CURSOR_HIDE)
            if not self._restore_registered:
                atexit.register(self._restore_cursor)
                self._restore_registered = True
            self._started = True
        self._clear_previous()
        self.stream.write("\n".join(lines))
        if restore_cursor:
            self.stream.write(self._CURSOR_SHOW)
        self.stream.write("\n")
        self.stream.flush()
        self._rendered_lines = self._physical_rows(lines, width)

    def cleanup(self) -> None:
        self._restore_cursor()

    def _restore_cursor(self) -> None:
        if not self.enabled or not self._started or self._finished:
            return
        self.stream.write(self._CURSOR_SHOW)
        self.stream.write("\n")
        self.stream.flush()
        self._finished = True

    @staticmethod
    def _terminal_width() -> int:
        return shutil.get_terminal_size(fallback=(80, 24)).columns

    @staticmethod
    def _physical_rows(lines: list[str], width: int) -> int:
        # A status line longer than the terminal width wraps onto multiple
        # physical rows. The cursor-up (`\x1b[NF`) and clear-line counts in
        # `_clear_previous` operate on physical rows, so they must account for
        # that wrapping; otherwise the redraw clears too few rows and leaves
        # stale, partially-overwritten copies of the status block behind.
        if width <= 0:
            return len(lines)
        return sum(max(1, -(-len(line) // width)) for line in lines)

    def _clear_previous(self) -> None:
        if self._rendered_lines == 0:
            return
        self.stream.write(f"\x1b[{self._rendered_lines}F")
        for _ in range(self._rendered_lines):
            self.stream.write(f"{self._CLEAR_LINE}\n")
        self.stream.write(f"\x1b[{self._rendered_lines}F")

    def _format_lines(self, state: ProgressRenderState) -> list[str]:
        games_completed = sum(worker.processed_games for worker in state.workers)
        games_completed = min(state.total_games, games_completed)
        samples = sum(worker.samples for worker in state.workers)
        eta = self._format_eta(
            state.elapsed_seconds, games_completed, state.total_games
        )
        header = " ".join(
            [
                self.label,
                f"status={state.status}",
                f"games={games_completed}/{state.total_games}",
                f"{self.unit_label}={samples}",
                f"{self.unit_label}/s={self._format_rate(samples, state.elapsed_seconds)}",
                f"elapsed={self._format_duration(state.elapsed_seconds)}",
                f"eta={eta}",
            ]
        )
        if state.message is not None:
            header = f"{header} {state.message}"
        total = " ".join(
            [
                f"{'total':<{self._ROW_LABEL_WIDTH}}",
                f"[{self._bar(games_completed, state.total_games)}]",
                self._percent(games_completed, state.total_games),
            ]
        )
        lines = [self._format_worker(worker) for worker in state.workers]
        lines.append(total)
        lines.extend(state.detail_lines)
        lines.append(header)
        return lines

    def _format_worker(self, worker: WorkerProgressState) -> str:
        game_range = self._game_range(worker.start_game, worker.total_games)
        return " ".join(
            [
                f"{f'w{worker.worker_id:02d}':<{self._ROW_LABEL_WIDTH}}",
                f"[{self._bar(worker.processed_games, worker.total_games)}]",
                f"games={worker.processed_games}/{worker.total_games}",
                f"{self.unit_label}={worker.samples}",
                f"range={game_range}",
            ]
        )

    def _bar(self, completed: int, total: int) -> str:
        if total <= 0:
            return "░" * self._BAR_WIDTH
        if completed >= total:
            filled = self._BAR_WIDTH
        else:
            filled = max(0, completed * self._BAR_WIDTH // total)
        return "█" * filled + "░" * (self._BAR_WIDTH - filled)

    @staticmethod
    def _percent(completed: int, total: int) -> str:
        if total <= 0:
            return "0.0%"
        return f"{completed / total:6.1%}"

    @staticmethod
    def _game_range(start_game: int, games: int) -> str:
        if games <= 0:
            return "none"
        return f"{start_game + 1}-{start_game + games}"

    @staticmethod
    def _format_rate(samples: int, elapsed_seconds: float) -> str:
        if elapsed_seconds <= 0:
            return "0.0"
        return f"{samples / elapsed_seconds:.1f}"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = int(seconds) if seconds > 0 else 0
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @classmethod
    def _format_eta(cls, elapsed_seconds: float, completed: int, total: int) -> str:
        if total <= 0 or completed >= total:
            return cls._format_duration(0.0)
        if completed <= 0 or elapsed_seconds <= 0:
            return "--:--:--"
        remaining = elapsed_seconds * (total - completed) / completed
        return cls._format_duration(remaining)


__all__ = [
    "AnsiProgressRenderer",
    "ProgressRenderState",
    "ProgressStatus",
    "WorkerProgressState",
]
