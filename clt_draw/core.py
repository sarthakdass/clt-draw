"""The mathematics of clt-draw, with no user interface.

You supply a curve (non-negative values on an evenly spaced grid). It is treated as a probability
density and discretised onto a *lattice* of ``LATTICE`` points, so that every random variable below
takes values k = 0, 1, ..., LATTICE - 1 with probabilities p[k].

Sums are never sampled. The probability mass function (pmf) of a sum of independent draws is the
convolution of the individual pmfs, so we compute it exactly. To save work we only ever double:

    pmf of n draws   =   (pmf of n/2 draws)  convolved with itself          n = 2, 4, 8, ..., 1024

which is 10 FFT convolutions in total instead of 1023 for n = 1024. Each pmf is then standardised
(mean 0, standard deviation 1) so that every n can be drawn on the same axes and compared with the
standard normal curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np

LATTICE = 1024           # points the drawn curve is resampled to
LEVELS = 10              # n = 2**1 ... 2**10
Z_MAX = 4.0              # the plot shows z in [-Z_MAX, Z_MAX]
Z_POINTS = 801           # samples of each density across that range
MIN_COLUMNS = 20         # a curve must cover at least this many of the drawing columns

Z_GRID = np.linspace(-Z_MAX, Z_MAX, Z_POINTS)


@dataclass
class Level:
    """The standardised sum of ``n`` independent draws from the user's density."""

    n: int
    density: np.ndarray      # density of Z_n at Z_GRID
    ks: float                # Kolmogorov distance to the standard normal, sup |F_n - Phi|
    skew: float              # skewness (the normal has 0)
    excess_kurtosis: float   # excess kurtosis (the normal has 0)


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z) ** 2) / np.sqrt(2.0 * np.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)."""
    z = np.asarray(z, dtype=float)
    x = np.abs(z) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf_abs = 1.0 - poly * np.exp(-x * x)
    return 0.5 * (1.0 + np.sign(z) * erf_abs)


def values_from_columns(columns: Sequence[float], lattice: int = LATTICE) -> Optional[np.ndarray]:
    """Turn what the user drew into densities on the lattice.

    ``columns`` has one entry per horizontal position, ``nan`` where nothing was drawn. The span between
    the first and last drawn column is the support; gaps inside it are filled in linearly, anything below
    the baseline counts as zero, and the result is resampled to ``lattice`` points. Returns ``None`` if
    there is not enough drawn to work with.
    """
    cols = np.asarray(columns, dtype=float)
    drawn = np.flatnonzero(~np.isnan(cols))
    if drawn.size < MIN_COLUMNS:
        return None
    first, last = int(drawn[0]), int(drawn[-1])
    idx = np.arange(first, last + 1)
    filled = np.interp(idx, drawn, cols[drawn])
    values = np.interp(np.linspace(0, filled.size - 1, lattice), np.arange(filled.size), filled)
    values = np.clip(values, 0.0, None)
    return values if values.sum() > 0 else None


def convolve_with_itself(pmf: np.ndarray) -> np.ndarray:
    """pmf of the sum of two independent copies, by FFT. Length 2*len - 1."""
    out_len = 2 * pmf.size - 1
    size = 1 << (out_len - 1).bit_length()
    spec = np.fft.rfft(pmf, size)
    out = np.fft.irfft(spec * spec, size)[:out_len]
    np.clip(out, 0.0, None, out=out)      # FFT round-off can leave -1e-18
    return out / out.sum()


def _standardised_level(pmf: np.ndarray, n: int, mean_idx: float, std_idx: float) -> Level:
    """Standardise the pmf of a sum of n draws and measure how normal it is."""
    k = np.arange(pmf.size)
    scale = std_idx * np.sqrt(n)                     # lattice units per unit of z
    z = (k - n * mean_idx) / scale

    density = np.interp(Z_GRID * scale + n * mean_idx, k, pmf, left=0.0, right=0.0) * scale

    cdf = np.cumsum(pmf)
    phi = normal_cdf(z)
    ks = float(max(np.abs(cdf - phi).max(), np.abs(cdf - pmf - phi).max()))   # both sides of each jump

    skew = float(np.dot(pmf, z ** 3))
    kurt = float(np.dot(pmf, z ** 4)) - 3.0
    return Level(n=n, density=density, ks=ks, skew=skew, excess_kurtosis=kurt)


def iter_levels(values: np.ndarray, levels: int = LEVELS) -> Iterator[Level]:
    """Yield the standardised sum for n = 1 (the curve itself), then n = 2, 4, ..., 2**levels.

    A generator, so a UI can draw each result as soon as it exists.
    """
    p = np.asarray(values, dtype=float)
    pmf = p / p.sum()
    k = np.arange(pmf.size)
    mean_idx = float(np.dot(k, pmf))
    std_idx = float(np.sqrt(np.dot((k - mean_idx) ** 2, pmf)))
    if std_idx == 0:
        raise ValueError("the curve has zero spread")

    yield _standardised_level(pmf, 1, mean_idx, std_idx)
    for lvl in range(1, levels + 1):
        pmf = convolve_with_itself(pmf)             # n/2 draws + n/2 draws  ->  n draws
        yield _standardised_level(pmf, 2 ** lvl, mean_idx, std_idx)


def compute_levels(values: np.ndarray, levels: int = LEVELS) -> list[Level]:
    return list(iter_levels(values, levels))
