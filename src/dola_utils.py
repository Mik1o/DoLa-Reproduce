"""Helpers for selecting and validating DoLa-related settings."""

from __future__ import annotations


def select_dola_layers(
    total_layers: int,
    strategy: str = "last_k",
    count: int = 4,
) -> list[int]:
    """Return a placeholder list of layer indices for future DoLa experiments."""
    if total_layers <= 0:
        raise ValueError("total_layers must be positive.")
    if count <= 0:
        raise ValueError("count must be positive.")

    # TODO: replace this placeholder heuristic with experiment-driven selection.
    start = max(total_layers - count, 0)
    return list(range(start, total_layers))


def validate_dola_layers(dola_layers: list[int] | None) -> list[int]:
    """Validate a user-provided DoLa layer list."""
    if dola_layers is None:
        return []
    if any(layer < 0 for layer in dola_layers):
        raise ValueError("dola_layers must contain non-negative integers.")
    return sorted(set(dola_layers))
