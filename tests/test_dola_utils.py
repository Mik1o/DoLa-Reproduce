"""Tests for minimal DoLa layer utility helpers."""

from __future__ import annotations

import pytest

from src.dola_utils import (
    describe_dola_pair,
    get_mature_layer_index,
    normalize_layer_bucket,
    validate_candidate_premature_layers,
    validate_mature_layer,
    validate_premature_layer,
)



def test_validate_premature_layer_accepts_valid_index() -> None:
    """A valid premature layer should pass without error."""
    validate_premature_layer(0, 4)
    validate_premature_layer(2, 4)



def test_validate_premature_layer_rejects_invalid_indices() -> None:
    """Invalid premature layers should raise clear errors."""
    with pytest.raises(ValueError, match="non-negative"):
        validate_premature_layer(-1, 4)
    with pytest.raises(ValueError, match="smaller than the mature final layer index"):
        validate_premature_layer(3, 4)



def test_validate_mature_layer_rejects_out_of_range_values() -> None:
    """The mature layer helper should guard invalid explicit layer configs."""
    validate_mature_layer(3, 4)
    with pytest.raises(ValueError, match="non-negative"):
        validate_mature_layer(-1, 4)
    with pytest.raises(ValueError, match="smaller than num_hidden_layers"):
        validate_mature_layer(4, 4)



def test_normalize_layer_bucket_preserves_first_seen_order() -> None:
    """Shifted or candidate buckets should dedupe while keeping stable order."""
    assert normalize_layer_bucket([5, 1, 5, 3], mature_layer=6, num_hidden_layers=8) == [5, 1, 3]



def test_validate_candidate_premature_layers_normalizes_valid_dynamic_list() -> None:
    """Dynamic DoLa candidate layers should stay stable and below mature."""
    assert validate_candidate_premature_layers([5, 1, 3], mature_layer=6, num_hidden_layers=8) == [5, 1, 3]



def test_validate_candidate_premature_layers_rejects_bad_inputs() -> None:
    """Dynamic DoLa layer lists should fail clearly on empty or invalid layers."""
    with pytest.raises(ValueError, match="at least one"):
        validate_candidate_premature_layers([], mature_layer=6, num_hidden_layers=8)
    with pytest.raises(ValueError, match="smaller than mature_layer"):
        validate_candidate_premature_layers([1, 6], mature_layer=6, num_hidden_layers=8)



def test_get_mature_layer_index_returns_last_decoder_layer() -> None:
    """The mature layer should be the final decoder block index."""
    assert get_mature_layer_index(4) == 3



def test_describe_dola_pair_returns_readable_text() -> None:
    """The layer pair description should be stable and compact."""
    assert describe_dola_pair(1, 3) == "premature_layer=1, mature_layer=3"
    with pytest.raises(ValueError, match="smaller than mature_layer"):
        describe_dola_pair(3, 3)
