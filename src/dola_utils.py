"""Helpers for minimal DoLa-style layer pairing and validation."""

from __future__ import annotations



def validate_premature_layer(premature_layer: int, num_hidden_layers: int) -> None:
    """Validate a zero-based premature layer index for DoLa-style scoring."""
    if num_hidden_layers <= 1:
        raise ValueError("num_hidden_layers must be greater than 1 for DoLa-style scoring.")
    if premature_layer < 0:
        raise ValueError("premature_layer must be a non-negative integer.")
    if premature_layer >= num_hidden_layers - 1:
        raise ValueError(
            "premature_layer must be smaller than the mature final layer index "
            f"({num_hidden_layers - 1})."
        )



def validate_mature_layer(mature_layer: int, num_hidden_layers: int) -> None:
    """Validate a zero-based mature layer index."""
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive.")
    if mature_layer < 0:
        raise ValueError("mature_layer must be a non-negative integer.")
    if mature_layer >= num_hidden_layers:
        raise ValueError(
            f"mature_layer must be smaller than num_hidden_layers ({num_hidden_layers})."
        )



def validate_candidate_premature_layers(
    candidate_premature_layers: list[int] | None,
    mature_layer: int,
    num_hidden_layers: int,
) -> list[int]:
    """Validate and normalize candidate premature layers for dynamic DoLa scoring."""
    validate_mature_layer(mature_layer, num_hidden_layers)
    if not candidate_premature_layers:
        raise ValueError("candidate_premature_layers must contain at least one layer index.")

    normalized_layers = sorted(set(int(layer) for layer in candidate_premature_layers))
    if len(normalized_layers) != len(candidate_premature_layers):
        raise ValueError("candidate_premature_layers must not contain duplicate layers.")
    if any(layer < 0 for layer in normalized_layers):
        raise ValueError("candidate_premature_layers must contain non-negative integers.")
    if any(layer >= num_hidden_layers for layer in normalized_layers):
        raise ValueError(
            "candidate_premature_layers must be smaller than num_hidden_layers "
            f"({num_hidden_layers})."
        )
    if any(layer >= mature_layer for layer in normalized_layers):
        raise ValueError("Every candidate premature layer must be smaller than mature_layer.")
    return normalized_layers



def get_mature_layer_index(num_hidden_layers: int) -> int:
    """Return the zero-based mature layer index for the model's final layer."""
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive.")
    return num_hidden_layers - 1



def describe_dola_pair(premature_layer: int, mature_layer: int) -> str:
    """Return a short text description of the selected DoLa layer pair."""
    if premature_layer < 0 or mature_layer < 0:
        raise ValueError("Layer indices must be non-negative integers.")
    if premature_layer >= mature_layer:
        raise ValueError("premature_layer must be smaller than mature_layer.")
    return f"premature_layer={premature_layer}, mature_layer={mature_layer}"



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
