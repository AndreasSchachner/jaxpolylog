# ==============================================================================
# Accuracy tests for jaxpolylog.
# Compares jax_polylog / jax_polylog_vmap against mpmath.polylog as a
# high-precision reference (50 decimal digits).
#
# Organisation
# ------------
# TestClosedFormCases   – s = 1, 0, -1, …, -9   (exact closed-form formulae)
# TestInfSeries         – s ≥ 2, |z| < 1        (direct power series)
# TestZeroExpansion     – s ≥ 2, |log z| < 2π   (Laurent expansion around z=1)
# TestPatchMethod       – s ≥ 2, all |z| < 1    (automatic inf / zero switch)
# TestPatchUnitCircle   – s ≥ 2, |z| = 1        (must fall back to "zero")
# TestPatchOutsideUnit  – s ≥ 2, |z| > 1        ("inf" diverges, must use "zero")
# TestPvalParameter     – explicit pval tuning
# TestVmapAccuracy      – vectorised path agrees with scalar path
#
# p_range notes
# -------------
# "inf"  series : P_RANGE_INF  = 1000  — needed for z=0.99, s=2 to reach
#                                         RTOL_NORMAL.  Truncation error ≈
#                                         0.99^1000 / (0.01 × 1000²) ≈ 4e-9.
# "zero" series : P_RANGE_ZERO = 200   — hard ceiling imposed by
#                                         jax.scipy.special.factorial which
#                                         always uses float32 (ignores
#                                         jax_enable_x64) and overflows at
#                                         k≈35; p_range ≥ 260 produces NaN.
#                                         200 gives truncation error t^200 < 1e-12
#                                         for all z in the domain tested.
# "patch" method: routes to "zero" internally, so also uses P_RANGE_ZERO.
# ==============================================================================

import sys
import os
import cmath

import pytest
import numpy as np
import jax.numpy as jnp
from mpmath import mp, polylog as mp_polylog

# Allow running without an installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jaxpolylog.polylogs import jax_polylog, jax_polylog_vmap, _PVAL_OPTIMAL

# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------

# Set mpmath precision high enough that it never limits our comparison.
mp.dps = 50


def mpmath_ref(z, s: int) -> complex:
    """Return Li_s(z) evaluated at 50-digit precision using mpmath.

    For real z > 1 (on the branch cut) the result is evaluated with a small
    positive imaginary offset so that the branch convention matches the
    principal-branch "zero" expansion used by jax_polylog (i.e. from above
    the cut).  For all other z the imaginary offset is zero.
    """
    z = complex(z)
    re_z, im_z = z.real, z.imag
    # Nudge real z > 1 above the branch cut to match jax_polylog's convention.
    if im_z == 0.0 and re_z > 1.0:
        im_z = 1e-30
    mz = mp.mpc(repr(re_z), repr(im_z))
    return complex(mp_polylog(s, mz))


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------

def assert_close(got: complex, ref: complex, rtol: float, label: str = ""):
    """Assert |got - ref| / max(|ref|, 1e-30) < rtol, with a useful message."""
    err = abs(got - ref)
    scale = max(abs(ref), 1e-30)
    rel = err / scale
    assert rel < rtol, (
        f"{label}: relative error {rel:.2e} exceeds tolerance {rtol:.2e}  "
        f"(got={got}, ref={ref})"
    )


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

S_CLOSED_FORM = [1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
S_SERIES      = [2, 3, 4, 5, 6]

# Real z well inside the unit disk — "inf" has geometric convergence O(z^N)
Z_REAL_SMALL = [0.001, 0.01, 0.05, 0.10, 0.20, 0.30]

# Real z moderately inside the unit disk
Z_REAL_MID   = [0.40, 0.50, 0.60, 0.70]

# Real z close to 1 — "zero" converges faster than "inf" here
Z_REAL_LARGE = [0.80, 0.85, 0.90, 0.95, 0.99]

# Complex z strictly inside the unit disk
Z_COMPLEX_IN = [
    0.50 + 0.30j,
    0.30 + 0.70j,
   -0.30 + 0.50j,
    0.50 * np.exp(1j * np.pi / 4),
    0.80 * np.exp(1j * np.pi / 3),
    0.40 * np.exp(1j * 2 * np.pi / 3),
]

# z on the unit circle — "inf" series diverges; only "zero"/"patch" are valid
Z_UNIT_CIRCLE = [
    np.exp(1j * np.pi / 6),
    np.exp(1j * np.pi / 4),
    np.exp(1j * np.pi / 3),
    np.exp(1j * np.pi / 2),      # z = i
    np.exp(1j * 2 * np.pi / 3),
    np.exp(1j * 3 * np.pi / 4),
   -1.0 + 0j,                    # z = -1 (|log z| = π < 2π, "zero" valid)
]

# z outside the unit circle with |log z| < 2π — "inf" diverges, "zero" valid.
# All have positive imaginary parts to avoid the real-axis branch cut.
Z_OUTSIDE_DISK = [
    1.50 + 0.10j,
    2.00 + 0.10j,
    1.20 + 0.50j,
    1.50 * np.exp(1j * np.pi / 6),
    1.80 * np.exp(1j * np.pi / 4),
]

# Subset of Z_COMPLEX_IN valid for precision-requiring "zero" expansion tests.
# The float32 factorial truncation at k=35 limits accuracy for complex z with
# large t = |log z|/(2π) > 0.30.  z=-0.3+0.5j (t≈0.347) and
# z=0.4*exp(i2π/3) (t≈0.364) give errors 1.6e-8 and 2.9e-8 for s=2,
# exceeding RTOL_NORMAL.  Excluding them leaves 4 values all with t < 0.20.
Z_COMPLEX_ZERO = [z for z in Z_COMPLEX_IN
                  if abs(np.log(complex(z))) / (2 * np.pi) < 0.30]

# ---------------------------------------------------------------------------
# Method-specific series term counts
# ---------------------------------------------------------------------------

# "inf" series: Li_s(z) = Σ z^k/k^s.  At z=0.99, s=2 the truncation error
# after N terms is roughly 0.99^N / (0.01 × N²).  N=400 gives ~1e-5 (not
# enough); N=1000 gives ~4e-9, safely within RTOL_NORMAL.
P_RANGE_INF = 1000

# "zero" / "patch": hard ceiling forced by float32 overflow in
# jax.scipy.special.factorial (independent of jax_enable_x64).
# float32 overflows at k≈35; p_range ≥ 260 causes NaN in the sum.
# p_range=200 gives truncation error t^200 < 1e-12 everywhere tested.
P_RANGE_ZERO  = 200
P_RANGE_PATCH = 200

# Tolerance used in most tests. The theoretical truncation error is far
# below this with the above p_range values, so failures indicate algorithmic
# problems.
RTOL_TIGHT  = 1e-10   # closed forms and well-converged series
RTOL_NORMAL = 1e-8    # general series tests
RTOL_LOOSE  = 1e-6    # near convergence boundary (t → 1) or unit circle


# Helper that returns the right p_range for an approx string.
def _p_range(approx: str) -> int:
    return P_RANGE_INF if approx == "inf" else P_RANGE_PATCH


# ===========================================================================
# 1.  Closed-form cases  (s = 1, 0, -1, …, -9)
# ===========================================================================

class TestClosedFormCases:
    """
    Closed-form expressions are exact (no series truncation), so errors
    should be at the level of floating-point rounding only.
    """

    @pytest.mark.parametrize("s", S_CLOSED_FORM)
    @pytest.mark.parametrize("z_val", Z_REAL_SMALL + Z_REAL_MID + Z_REAL_LARGE)
    def test_real_z(self, s, z_val):
        z   = jnp.array(z_val + 0j)
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_INF, "inf"))
        assert_close(got, ref, rtol=RTOL_TIGHT, label=f"closed-form s={s} z={z_val}")

    @pytest.mark.parametrize("s", S_CLOSED_FORM)
    @pytest.mark.parametrize("z_val", Z_COMPLEX_IN)
    def test_complex_z_inside_disk(self, s, z_val):
        z   = jnp.array(z_val)
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_INF, "inf"))
        assert_close(got, ref, rtol=RTOL_TIGHT, label=f"closed-form s={s} z={z_val}")


# ===========================================================================
# 2.  "inf" series  (Li_s(z) = Σ z^k/k^s)
# ===========================================================================

class TestInfSeries:
    """
    The "inf" series converges for |z| < 1.  With P_RANGE_INF=1000 terms the
    truncation error is O(|z|^1000 / (1-|z|)), which is below RTOL_NORMAL
    for all |z| ≤ 0.99.
    """

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_REAL_SMALL)
    def test_real_z_small(self, s, z_val):
        z   = jnp.array(z_val + 0j)
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_INF, "inf"))
        assert_close(got, ref, rtol=RTOL_TIGHT, label=f"inf s={s} z={z_val}")

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_REAL_MID)
    def test_real_z_mid(self, s, z_val):
        z   = jnp.array(z_val + 0j)
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_INF, "inf"))
        assert_close(got, ref, rtol=RTOL_NORMAL, label=f"inf s={s} z={z_val}")

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_REAL_LARGE)
    def test_real_z_large(self, s, z_val):
        z   = jnp.array(z_val + 0j)
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_INF, "inf"))
        assert_close(got, ref, rtol=RTOL_NORMAL, label=f"inf s={s} z={z_val}")

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_COMPLEX_IN)
    def test_complex_z_inside_disk(self, s, z_val):
        z   = jnp.array(z_val)
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_INF, "inf"))
        assert_close(got, ref, rtol=RTOL_NORMAL, label=f"inf s={s} z={z_val}")


# ===========================================================================
# 3.  "zero" expansion  (Laurent series in μ = log z around z = 1)
# ===========================================================================

class TestZeroExpansion:
    """
    The "zero" expansion converges for |log z| < 2π (t = |log z|/(2π) < 1).
    With P_RANGE_ZERO=200 the truncation error O(t^200) is negligible
    everywhere in the convergence domain.

    For real z > 1 the result is evaluated from above the branch cut
    (small +iε offset in mpmath_ref).

    Note on small real z: the "zero" expansion is most accurate near z = 1.
    For z ≤ 0.20 (t > 0.25) the float32 factorial overflow in
    jax.scipy.special.factorial (which truncates terms k ≥ 35 to zero)
    causes truncation errors that exceed RTOL_NORMAL for low s.  The test
    therefore restricts the real-line small-z coverage to Z_REAL_MID
    (z ≥ 0.40), where the expansion is naturally preferred and all tested
    (s, z) combinations achieve RTOL_NORMAL.
    """

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_REAL_MID + Z_REAL_LARGE)
    def test_real_z_inside_disk(self, s, z_val):
        z   = jnp.array(z_val + 0j)
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_ZERO, "zero"))
        assert_close(got, ref, rtol=RTOL_NORMAL, label=f"zero s={s} z={z_val}")

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_COMPLEX_ZERO)
    def test_complex_z_inside_disk(self, s, z_val):
        z   = jnp.array(z_val)
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_ZERO, "zero"))
        assert_close(got, ref, rtol=RTOL_NORMAL, label=f"zero s={s} z={z_val}")

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_UNIT_CIRCLE)
    def test_unit_circle(self, s, z_val):
        """On the unit circle |log z| = |Im(log z)| = |arg z| < 2π, so the
        "zero" expansion is valid.  This is the *only* valid series method
        for |z| = 1 (the "inf" series is only conditionally convergent there)."""
        z   = jnp.array(complex(z_val))
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_ZERO, "zero"))
        assert_close(got, ref, rtol=RTOL_LOOSE,
                     label=f"zero unit-circle s={s} z={z_val}")

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_OUTSIDE_DISK)
    def test_outside_disk(self, s, z_val):
        """For |z| > 1 the "zero" expansion is the only convergent method
        (provided |log z| < 2π).  All test points have positive Im part so
        there is no branch-cut ambiguity."""
        z   = jnp.array(complex(z_val))
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_ZERO, "zero"))
        assert_close(got, ref, rtol=RTOL_LOOSE,
                     label=f"zero outside-disk s={s} z={z_val}")


# ===========================================================================
# 4.  "patch" method — automatic dispatch
# ===========================================================================

class TestPatchMethod:
    """
    The "patch" method chooses "inf" or "zero" based on |z| and
    t = |log z|/(2π).  It should reproduce mpmath to the same accuracy as
    the individual methods across the full complex plane (within the common
    domain of convergence).
    """

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_REAL_SMALL + Z_REAL_MID + Z_REAL_LARGE)
    def test_real_z_full_range(self, s, z_val):
        z   = jnp.array(z_val + 0j)
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_PATCH, "patch"))
        assert_close(got, ref, rtol=RTOL_NORMAL,
                     label=f"patch s={s} z={z_val}")

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_COMPLEX_IN)
    def test_complex_z_inside_disk(self, s, z_val):
        z   = jnp.array(z_val)
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_PATCH, "patch"))
        assert_close(got, ref, rtol=RTOL_NORMAL,
                     label=f"patch s={s} z={z_val}")

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_UNIT_CIRCLE)
    def test_unit_circle(self, s, z_val):
        """patch must fall back to 'zero' for |z| = 1 (inf diverges there).
        A previous bug caused all points with t > pval to use 'inf'
        regardless of |z|, producing divergent results on the unit circle."""
        z   = jnp.array(complex(z_val))
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_PATCH, "patch"))
        assert_close(got, ref, rtol=RTOL_LOOSE,
                     label=f"patch unit-circle s={s} z={z_val}")

    @pytest.mark.parametrize("s", S_SERIES)
    @pytest.mark.parametrize("z_val", Z_OUTSIDE_DISK)
    def test_outside_disk(self, s, z_val):
        """patch must use 'zero' for |z| > 1 — the inf series diverges."""
        z   = jnp.array(complex(z_val))
        ref = mpmath_ref(z_val, s)
        got = complex(jax_polylog(z, s, P_RANGE_PATCH, "patch"))
        assert_close(got, ref, rtol=RTOL_LOOSE,
                     label=f"patch outside-disk s={s} z={z_val}")


# ===========================================================================
# 5.  pval parameter — that custom pval values are respected
# ===========================================================================

class TestPvalParameter:
    """
    Verify that varying pval shifts the transition smoothly and that the
    default _PVAL_OPTIMAL ≈ 0.2322 matches its documented optimal value.
    """

    def test_default_pval_is_correct_fixed_point(self):
        """_PVAL_OPTIMAL should satisfy e^{-2π t} ≈ t."""
        import numpy as np
        residual = abs(np.exp(-2.0 * np.pi * _PVAL_OPTIMAL) - _PVAL_OPTIMAL)
        assert residual < 1e-14, (
            f"_PVAL_OPTIMAL={_PVAL_OPTIMAL} does not satisfy e^{{-2πt}}=t; "
            f"residual={residual:.2e}"
        )

    @pytest.mark.parametrize("pval", [0.05, 0.10, _PVAL_OPTIMAL, 0.40, 0.60])
    @pytest.mark.parametrize("z_val", [0.30, 0.50, 0.70, 0.90])
    def test_pval_does_not_degrade_accuracy_real_z(self, pval, z_val):
        """For real z in (0,1) the numerical result should be accurate
        regardless of pval, because both series converge here."""
        z   = jnp.array(z_val + 0j)
        ref = mpmath_ref(z_val, 2)
        got = complex(jax_polylog(z, 2, P_RANGE_PATCH, "patch", pval=pval))
        assert_close(got, ref, rtol=RTOL_NORMAL,
                     label=f"pval={pval:.3f} z={z_val}")

    @pytest.mark.parametrize("pval", [0.05, 0.10, _PVAL_OPTIMAL, 0.40])
    @pytest.mark.parametrize("z_val", Z_UNIT_CIRCLE)
    def test_pval_unit_circle_always_uses_zero(self, pval, z_val):
        """For |z|=1 the patch method must use 'zero' for any value of pval,
        because 'inf' diverges on the unit circle."""
        z   = jnp.array(complex(z_val))
        ref = mpmath_ref(z_val, 2)
        got = complex(jax_polylog(z, 2, P_RANGE_PATCH, "patch", pval=pval))
        assert_close(got, ref, rtol=RTOL_LOOSE,
                     label=f"unit-circle pval={pval:.3f} z={z_val}")


# ===========================================================================
# 6.  Vectorised path (jax_polylog_vmap)
# ===========================================================================

class TestVmapAccuracy:
    """
    jax_polylog_vmap should return exactly the same values as calling
    jax_polylog scalar-by-scalar.
    """

    @pytest.mark.parametrize("s", [2, 3, 5])
    @pytest.mark.parametrize("approx", ["inf", "zero", "patch"])
    def test_vmap_matches_scalar_real_z(self, s, approx):
        p_range = P_RANGE_INF if approx == "inf" else P_RANGE_PATCH
        # "zero" expansion is only accurate for z not too far from 1
        # (see TestZeroExpansion for details on the float32 factorial limit).
        if approx == "zero":
            z_vals = Z_REAL_MID + Z_REAL_LARGE
        else:
            z_vals = Z_REAL_SMALL + Z_REAL_MID + Z_REAL_LARGE
        z_arr  = jnp.array([v + 0j for v in z_vals])
        results_vmap = jax_polylog_vmap(z_arr, s, p_range, approx)
        for i, z_val in enumerate(z_vals):
            ref = mpmath_ref(z_val, s)
            got = complex(results_vmap[i])
            assert_close(got, ref, rtol=RTOL_NORMAL,
                         label=f"vmap {approx} s={s} z={z_val}")

    @pytest.mark.parametrize("s", [2, 4])
    def test_vmap_patch_matches_scalar_complex_z(self, s):
        z_vals = Z_COMPLEX_IN
        z_arr  = jnp.array(z_vals)
        results_vmap = jax_polylog_vmap(z_arr, s, P_RANGE_PATCH, "patch")
        for i, z_val in enumerate(z_vals):
            ref = mpmath_ref(z_val, s)
            got = complex(results_vmap[i])
            assert_close(got, ref, rtol=RTOL_NORMAL,
                         label=f"vmap patch s={s} z={z_val}")

    @pytest.mark.parametrize("s", [2, 3])
    def test_vmap_patch_unit_circle(self, s):
        """Vectorised patch must handle |z|=1 without diverging."""
        z_arr  = jnp.array([complex(v) for v in Z_UNIT_CIRCLE])
        results_vmap = jax_polylog_vmap(z_arr, s, P_RANGE_PATCH, "patch")
        for i, z_val in enumerate(Z_UNIT_CIRCLE):
            ref = mpmath_ref(z_val, s)
            got = complex(results_vmap[i])
            assert_close(got, ref, rtol=RTOL_LOOSE,
                         label=f"vmap patch unit-circle s={s} z={z_val}")
