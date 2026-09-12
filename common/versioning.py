"""Shared output-versioning helper, used by every top-level exp_*.py script
so that different runs never silently overwrite each other's results.
"""

from __future__ import annotations

import re
from pathlib import Path


def next_version_dir(base_dir: Path) -> Path:
    """First unused ``<base_dir>/v<N>`` folder (v0, v1, v2, ...).

    Deliberately paranoid: this folder can live on a synced/bridged mount
    where a directory listing can occasionally be stale, so we don't trust
    ``iterdir()`` alone. We also keep a persistent ``.version_counter`` file
    that only ever increases, and refuse to reuse a v<N> directory that
    already has files in it (bumping to the next number instead). This
    guarantees we never silently overwrite a previous run's results.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    counter_file = base_dir / ".version_counter"

    scanned = [
        int(m.group(1))
        for p in base_dir.iterdir()
        if p.is_dir() and (m := re.fullmatch(r"v(\d+)", p.name))
    ]
    next_n = max(scanned, default=-1) + 1

    if counter_file.exists():
        try:
            next_n = max(next_n, int(counter_file.read_text().strip()))
        except ValueError:
            pass

    while (base_dir / f"v{next_n}").exists() and any((base_dir / f"v{next_n}").iterdir()):
        next_n += 1

    counter_file.write_text(str(next_n + 1))
    return base_dir / f"v{next_n}"


def latest_version_dir(base_dir: Path) -> Path | None:
    """The most recently created ``<base_dir>/v<N>`` folder that actually has
    files in it (the read-side counterpart to ``next_version_dir``'s
    writing-side logic -- "latest" means highest N among non-empty v<N>
    folders). Returns ``None`` if ``base_dir`` doesn't exist yet or has no
    such folder, so callers can decide how to handle "nothing to compare
    against yet" themselves."""
    if not base_dir.is_dir():
        return None
    candidates = [
        (int(m.group(1)), p)
        for p in base_dir.iterdir()
        if p.is_dir() and (m := re.fullmatch(r"v(\d+)", p.name)) and any(p.iterdir())
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
