from __future__ import annotations

import typing as tp


class _NameRegistry:
    """Global registry that ensures every dataset name is unique."""

    _seen: tp.ClassVar[tp.Set[str]] = set()
    _counters: tp.ClassVar[tp.Dict[str, int]] = {}

    @classmethod
    def register(cls, name: tp.Optional[str], prefix: str) -> str:
        """
        Register *name* and return the final unique name.

        If *name* is None   -> auto-generate from *prefix* + counter.
        If *name* is taken  -> raise ValueError immediately.
        """
        if name is None:
            base = prefix
            idx = cls._counters.get(base, 0)
            while f"{base}_{idx}" in cls._seen:
                idx += 1
            name = f"{base}_{idx}"
            cls._counters[base] = idx + 1

        if name in cls._seen:
            raise ValueError(
                f"Dataset name '{name}' is already in use. "
                "Provide a unique name explicitly."
            )
        cls._seen.add(name)
        return name

    @classmethod
    def reset(cls) -> None:
        """Clear registry (useful in tests)."""
        cls._seen.clear()
        cls._counters.clear()
