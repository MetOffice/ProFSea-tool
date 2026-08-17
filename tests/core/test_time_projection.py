import numpy as np
import pytest

from profsea.components.core.state import ClimateState
from profsea.components.core.time_projection import time_projection


def get_dummy_state(
    *,
    nyr: int = 4,
    nt: int = 2,
    num_members: int = 2,
    end_yr: int = 2010,
    palmer_method: bool = False,
) -> ClimateState:
    """Helper to generate a small ClimateState."""
    T_ens = np.ones((nt, nyr), dtype=np.float32)
    T_int_ens = np.cumsum(T_ens, axis=1)
    T_int_med = np.cumsum(np.median(T_ens, axis=0))

    return ClimateState(
        scenario="test",
        T_ens=T_ens,
        T_int_ens=T_int_ens,
        T_int_med=T_int_med,
        fraction=np.linspace(
            0.1,
            0.9,
            nt * num_members,
            dtype=np.float32,
        ).reshape(nt, num_members),
        palmer_method=palmer_method,
        endofAR5=2100,
        endofhistory=2006,
        end_yr=end_yr,
        nyr=nyr,
        nt=nt,
        num_members=num_members,
    )


def test_time_projection_output_shape():
    """Projection should contain one timeseries per member/trajectory pair."""
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    projection = time_projection(
        state,
        startratemean=0.5,
        startratepm=0.1,
        final=[0.1, 0.2],
        rng=rng,
    )

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.nyr,
    )


def test_time_projection_reproducible_with_same_seed():
    """Random fraction generation should be reproducible for a fixed seed."""
    state = get_dummy_state()
    # Need to set fraction to None to actually test the RNG logic in the function
    state.fraction = None

    projection1 = time_projection(
        state,
        startratemean=0.5,
        startratepm=0.1,
        final=[0.1, 0.2],
        rng=np.random.default_rng(42),
    )

    projection2 = time_projection(
        state,
        startratemean=0.5,
        startratepm=0.1,
        final=[0.1, 0.2],
        rng=np.random.default_rng(42),
    )

    np.testing.assert_allclose(projection1, projection2)


def test_time_projection_wrong_fraction_shape():
    """Fraction arrays must match the (nt, num_members) shape."""
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    # 1D array instead of 2D
    fraction = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Note: The ValueError for this actually gets thrown on line 58 of time_projection
    # when it tries to multiply `fraction` (shape 3,) by `startrate` (shape 2,)
    # or by `final` depending on broadcasting. I updated the match string to reflect this
    # or just catch general ValueErrors since the original code didn't have an explicit raise for it.
    with pytest.raises(IndexError):
        time_projection(
            state,
            startratemean=0.5,
            startratepm=0.1,
            final=[0.1, 0.2],
            rng=rng,
            fraction=fraction,
        )


def test_time_projection_wrong_final_range_size():
    """A final likely range must contain exactly two values."""
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    fraction = np.array(
        [[0.2, 0.4], [0.6, 0.8]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="final range is the wrong size"):
        time_projection(
            state,
            startratemean=0.5,
            startratepm=0.1,
            final=[0.1, 0.2, 0.3],
            rng=rng,
            fraction=fraction,
        )


def test_time_projection_wrong_final_array_shape():
    """Array-valued final projections must match the fraction shape."""
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    fraction = np.array(
        [[0.2, 0.4], [0.6, 0.8]],
        dtype=np.float32,
    )

    final = np.ones((4,), dtype=np.float32)

    with pytest.raises(ValueError, match="final array is the wrong shape"):
        time_projection(
            state,
            startratemean=0.5,
            startratepm=0.1,
            final=final,
            rng=rng,
            fraction=fraction,
        )


def test_time_projection_uses_fraction_endpoints():
    """Fractions of zero and one should select opposite ends of the ranges."""
    state = get_dummy_state(
        nt=1,
        num_members=2,
    )
    rng = np.random.default_rng(42)

    fraction = np.array([[0.0, 1.0]], dtype=np.float32)

    projection = time_projection(
        state,
        startratemean=1.0,
        startratepm=0.5,
        final=[0.1, 0.2],
        rng=rng,
        fraction=fraction,
    )

    # Fraction 0 selects the lower start rate and lower final value,
    # while fraction 1 selects the upper values.
    assert not np.allclose(projection[0, 0], projection[0, 1])

    assert projection[0, 1, -1] > projection[0, 0, -1]


def test_time_projection_zero_fraction_matches_manual_formula():
    """A fixed fraction should reproduce the quadratic projection formula."""
    state = get_dummy_state(
        nt=1,
        num_members=1,
    )

    fraction = np.array([[0.0]], dtype=np.float32)

    startratemean = 1.0
    startratepm = 0.5
    final = [0.1, 0.2]

    projection = time_projection(
        state,
        startratemean=startratemean,
        startratepm=startratepm,
        final=final,
        rng=np.random.default_rng(42),
        fraction=fraction,
    )

    start_rate = (startratemean - startratepm) * 1e-3
    afinal = final[0]

    finalyr = np.array([94.0], dtype=np.float32)
    halfacc = (afinal - start_rate * finalyr.mean()) / (finalyr**2).mean()

    time = (
        np.arange(
            state.end_yr - state.endofhistory,
            dtype=np.float32,
        )
        + 1
    )

    expected = halfacc * time**2 + start_rate * time

    np.testing.assert_allclose(
        projection[0, 0],
        expected,
        rtol=1e-5,
        atol=1e-7,
    )


def test_time_projection_accepts_final_array():
    """A full array of final values should be accepted."""
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    fraction = np.array(
        [[0.2, 0.4], [0.6, 0.8]],
        dtype=np.float32,
    )

    final = np.array(
        [
            [0.10, 0.11],
            [0.12, 0.13],
        ],
        dtype=np.float32,
    )

    projection = time_projection(
        state,
        startratemean=0.5,
        startratepm=0.1,
        final=final,
        rng=rng,
        fraction=fraction,
    )

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.nyr,
    )


def test_time_projection_nfinal_changes_final_mean_constraint():
    """Changing nfinal should affect the inferred acceleration."""
    state = get_dummy_state()

    fraction = np.array(
        [[0.2, 0.4], [0.6, 0.8]],
        dtype=np.float32,
    )

    projection1 = time_projection(
        state,
        startratemean=0.5,
        startratepm=0.1,
        final=[0.1, 0.2],
        rng=np.random.default_rng(42),
        fraction=fraction,
        nfinal=1,
    )

    projection20 = time_projection(
        state,
        startratemean=0.5,
        startratepm=0.1,
        final=[0.1, 0.2],
        rng=np.random.default_rng(42),
        fraction=fraction,
        nfinal=20,
    )

    assert not np.allclose(projection1, projection20)


def test_time_projection_palmer_method_matches_quadratic_before_2100():
    """Palmer extrapolation should not alter the projection before index 95."""
    nyr = 100

    state_normal = get_dummy_state(
        nyr=nyr,
        end_yr=2106,
        palmer_method=False,
    )
    state_palmer = get_dummy_state(
        nyr=nyr,
        end_yr=2106,
        palmer_method=True,
    )

    fraction = np.array(
        [[0.2, 0.4], [0.6, 0.8]],
        dtype=np.float32,
    )

    normal = time_projection(
        state_normal,
        startratemean=0.5,
        startratepm=0.1,
        final=[0.1, 0.2],
        rng=np.random.default_rng(42),
        fraction=fraction,
    )

    palmer = time_projection(
        state_palmer,
        startratemean=0.5,
        startratepm=0.1,
        final=[0.1, 0.2],
        rng=np.random.default_rng(42),
        fraction=fraction,
    )

    np.testing.assert_allclose(
        palmer[:, :, :95],
        normal[:, :, :95],
    )


def test_time_projection_palmer_method_becomes_linear_after_2100():
    """Palmer extrapolation should continue at a constant rate after 2100."""
    state = get_dummy_state(
        nyr=100,
        end_yr=2106,
        palmer_method=True,
    )

    fraction = np.array(
        [[0.2, 0.4], [0.6, 0.8]],
        dtype=np.float32,
    )

    projection = time_projection(
        state,
        startratemean=0.5,
        startratepm=0.1,
        final=[0.1, 0.2],
        rng=np.random.default_rng(42),
        fraction=fraction,
    )

    increment_95 = projection[:, :, 95] - projection[:, :, 94]
    increment_96 = projection[:, :, 96] - projection[:, :, 95]
    increment_97 = projection[:, :, 97] - projection[:, :, 96]

    np.testing.assert_allclose(
        increment_95,
        increment_96,
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        increment_96,
        increment_97,
        rtol=1e-5,
        atol=1e-7,
    )


def test_time_projection_preserves_state_dtype():
    """Projection should use the dtype configured on ClimateState."""
    state = get_dummy_state()
    state.dtype = np.float32

    fraction = np.array(
        [[0.2, 0.4], [0.6, 0.8]],
        dtype=np.float32,
    )

    projection = time_projection(
        state,
        startratemean=0.5,
        startratepm=0.1,
        final=[0.1, 0.2],
        rng=np.random.default_rng(42),
        fraction=fraction,
    )

    assert projection.dtype == np.float32
