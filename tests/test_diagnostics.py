import numpy as np
import pytest
from malleable.llm.diagnostics import _captured_reference, comparison, diagnose


def residual_reference(x):
    for i in range(2):
        x = x + 1
        x = x + 2
    return x


def incomplete_reference(x):
    x = x + 1
    return x


def test_residual_observation_preserves_math_and_copies_values():
    captured = {}
    observed = _captured_reference(residual_reference, captured)
    x = np.array([1., 2.])
    assert np.array_equal(observed(x), residual_reference(x))
    assert len(captured) == 4
    assert np.array_equal(captured['attention_residual', 0, None], x + 1)
    assert np.array_equal(captured['mlp_residual', 1, None], x + 6)
    with pytest.raises(ValueError, match='instrumentation changed'):
        _captured_reference(incomplete_reference, {})


def test_diagnostic_cannot_search_held_out_or_unbounded_targets():
    with pytest.raises(ValueError, match='validation-only'):
        diagnose('unused', 'unused', split='held-out')
    for limit in (0, 129, True, 1.5):
        with pytest.raises(ValueError, match='1..128'):
            diagnose('unused', 'unused', limit_targets=limit)


def test_comparison_reports_localized_difference():
    result = comparison(np.array([1., 3.]), np.array([1., 2.]))
    assert result['max_absolute_error'] == 1
    assert result['relative_l2_error'] == pytest.approx(1 / np.sqrt(5))
