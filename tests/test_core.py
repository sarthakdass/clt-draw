import numpy as np
import pytest

from clt_draw import (LATTICE, LEVELS, Z_GRID, compute_levels, normal_cdf, normal_pdf,
                      values_from_columns)
from clt_draw.core import convolve_with_itself


def lumpy(m=LATTICE):
    """A deliberately un-normal curve: two sharp bumps and a ramp."""
    x = np.linspace(0, 1, m)
    return np.exp(-((x - 0.2) / 0.03) ** 2) + 0.6 * np.exp(-((x - 0.75) / 0.05) ** 2) + 0.3 * x


def test_normal_cdf_matches_known_values():
    assert normal_cdf(0.0) == pytest.approx(0.5, abs=1e-7)
    assert normal_cdf(1.959964) == pytest.approx(0.975, abs=2e-7)
    assert normal_cdf(-1.0) == pytest.approx(0.158655254, abs=2e-7)


def test_doubling_matches_direct_repeated_convolution():
    p = lumpy(64)
    p = p / p.sum()
    direct4 = np.convolve(np.convolve(np.convolve(p, p), p), p)      # four draws, the slow way
    doubled4 = convolve_with_itself(convolve_with_itself(p))         # 2 then 2+2
    assert np.allclose(direct4, doubled4, atol=1e-12)


def test_two_uniform_draws_give_the_triangle():
    levels = compute_levels(np.ones(LATTICE), levels=1)
    peak = levels[1].density.max()
    # Two uniforms, standardized, make a triangle on [-sqrt(6), sqrt(6)] whose peak is 1/sqrt(6).
    assert peak == pytest.approx(1 / np.sqrt(6), rel=2e-3)
    assert levels[1].density[0] == 0.0 and levels[1].density[-1] == 0.0


def test_levels_cover_powers_of_two():
    levels = compute_levels(lumpy())
    assert [lv.n for lv in levels] == [1] + [2 ** i for i in range(1, LEVELS + 1)]


def test_every_standardized_density_integrates_to_one():
    for lv in compute_levels(lumpy()):
        area = np.sum((lv.density[1:] + lv.density[:-1]) / 2 * np.diff(Z_GRID))   # trapezoid rule
        assert area == pytest.approx(1.0, abs=0.02)


def test_skewness_and_kurtosis_scale_exactly():
    levels = compute_levels(lumpy())
    s1, k1 = levels[0].skew, levels[0].kurtosis
    for lv in levels[1:]:
        # exact in theory (cumulants add); the absolute slack covers FFT round-off once the values are ~1e-3
        assert lv.skew == pytest.approx(s1 / np.sqrt(lv.n), rel=1e-5, abs=1e-7)
        assert lv.kurtosis - 3 == pytest.approx((k1 - 3) / lv.n, rel=1e-5, abs=1e-7)


def test_distance_to_normal_shrinks_and_ends_small():
    ks = [lv.ks for lv in compute_levels(lumpy())]
    assert ks[0] > 0.1                       # the drawn curve is nothing like a normal
    assert ks[-1] < 0.01                     # 1024 draws: essentially there
    assert all(b < a for a, b in zip(ks[:-1], ks[1:]))


def test_normal_input_stays_normal():
    x = np.linspace(-6, 6, LATTICE)
    levels = compute_levels(np.exp(-0.5 * x ** 2))
    assert max(lv.ks for lv in levels) < 0.01
    assert np.allclose(levels[-1].density, normal_pdf(Z_GRID), atol=2e-3)


def test_values_from_columns_fills_gaps_and_clips():
    cols = np.full(100, np.nan)
    cols[10:30] = 1.0
    cols[60:80] = 3.0
    cols[15] = -2.0                           # below the baseline
    v = values_from_columns(cols)
    assert v.size == LATTICE and v.min() >= 0
    assert v.max() == pytest.approx(3.0)      # support runs 10..79, so nothing was added outside it


def test_too_little_drawn_is_rejected():
    assert values_from_columns(np.full(100, np.nan)) is None
    cols = np.full(100, np.nan)
    cols[:5] = 1.0
    assert values_from_columns(cols) is None
    assert values_from_columns(np.zeros(100)) is None
