from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from profsea.components.core.global_model import ClimateState


def time_projection(
    state: ClimateState,
    startratemean: float,
    startratepm: float,
    final,
    rng: np.random.Generator,
    nfinal: int = 1,
    fraction: np.ndarray = None,
) -> np.ndarray:
    """Project a quantity which is a quadratic function of time.

    Parameters
    ----------
    startratemean: float
        Rate of GMSLR at the start in mm yr-1.
    startratepm: float
        Start rate error in mm yr-1.
    final: list | np.ndarray
        Likely range in m for GMSLR at the end of AR5.
    nfinal: int
        Number of years at the end over which final is a time-mean.
    fraction: np.ndarray
        Random numbers in the range 0 to 1.

    Returns
    -------
    np.ndarray
        Projection of the quantity.
    """
    # Create a field of elapsed time since start in years
    timeendofAR5 = state.endofAR5 - state.endofhistory + 1
    time = (np.arange(state.end_yr - state.endofhistory, dtype=state.dtype) + 1)

    if fraction is None:
        fraction = rng.random((state.num_members, state.nt), dtype=state.dtype)
    elif fraction.size != state.num_members * state.nt:
        raise ValueError("fraction is the wrong size")

    fraction = fraction.reshape(state.num_members, state.nt)

    # Convert inputs to startrate (m yr-1) and afinal (m), where both are
    # arrays with the size of fraction
    startrate = ((
        startratemean + startratepm * np.array([-1, 1])
    ) * 1e-3).astype(state.dtype)  # convert mm yr-1 to m yr-1
    finalisrange = isinstance(final, Sequence)

    if finalisrange:
        if len(final) != 2:
            raise ValueError("final range is the wrong size")
        afinal = (1 - fraction) * final[0] + fraction * final[1]
    else:
        if final.shape != fraction.shape:
            raise ValueError("final array is the wrong shape")
        afinal = final

    startrate = (1 - fraction) * startrate[0] + fraction * startrate[1]

    # For terms where the rate increases linearly in time t, we can write GMSLR as
    #   S(t) = a*t**2 + b*t
    # where a is 0.5*acceleration and b is start rate. Hence
    #   a = S/t**2-b/t = (S-b*t)/t**2
    # If nfinal=1, the following two lines are equivalent to
    # halfacc=(final-startyr*nyr)/nyr**2
    finalyr = np.arange(nfinal, dtype=state.dtype) - nfinal + 94 + 1  # last element ==nyr
    halfacc = (afinal - startrate * finalyr.mean()) / (finalyr**2).mean()
    quadratic = halfacc[:, :, np.newaxis] * (time**2)
    linear = startrate[:, :, np.newaxis] * time

    # If acceleration ceases for t>t0, the rate is 2*a*t0+b thereafter, so
    #   S(t) = a*t0**2 + b*t0 + (2*a*t0+b)*(t-t0)
    #        = a*t0*(2*t - t0) + b*t
    # i.e. the quadratic term is replaced, the linear term unaffected
    # The quadratic also = a*t**2-a*(t-t0)**2

    if state.palmer_method:
        y = halfacc[:, :, np.newaxis] * timeendofAR5 * ((2 * time) - timeendofAR5)
        quadratic[:, :, 95:] = y[:, :, 95:]

    quadratic += linear

    quadratic = quadratic.reshape(
        quadratic.shape[0] * quadratic.shape[1], quadratic.shape[2]
    )

    return quadratic
