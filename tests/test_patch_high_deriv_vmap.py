# ==============================================================================
# Regression tests for the v0.3.0 fixes to ``approx="patch"``.
#
# Pins the three guarantees that v0.3.0 ships:
#
#   Fix 1 (double-where dispatch):  the ``patch`` branch must NEVER return
#         NaN for ``|z| ≤ 1``, at any ``p_range``.  Previously the
#         always-evaluated ``"zero"`` branch overflowed at small ``|z|`` and
#         poisoned the result via ``0 · inf = NaN``.
#
#   Fix 2 (stable zero coeffs):  ``"zero"`` (and patch-internal "zero") must
#         work at any ``p_range`` without overflowing the Bernoulli table.
#         Previously ``jax.scipy.special.bernoulli(p_range)`` overflowed
#         fp64 for ``p_range ≳ 250``.
#
#   Fix 3 (Li_1/z stable):  high-order autodiff (n ≥ 2) of ``Li_2(z)`` at
#         small ``|z|`` must remain at fp64 precision instead of suffering
#         the catastrophic cancellation of the literal ``-log1p(-z)/z``.
#
# Each guarantee is pinned at a specific test case taken from the
# v0.3.0 acceptance sweep (see commit message).
# ==============================================================================

import sys
import os

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from mpmath import mp, polylog as mp_polylog, diff as mp_diff

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jaxpolylog.polylogs import jax_polylog, jax_polylog_vmap, _PVAL_OPTIMAL

mp.dps = 50
PVAL = _PVAL_OPTIMAL


def _nth_holomorphic_deriv(f, n):
    """Return ``g(z) = d^n f / dz^n`` via nested forward-mode JVPs."""
    if n == 0:
        return f
    inner = _nth_holomorphic_deriv(f, n - 1)
    def g(z):
        _, t = jax.jvp(inner, (z,), (jnp.ones_like(z),))
        return t
    return g


def _relerr(a, b):
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    a, b = complex(a), complex(b)
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ----------------------------------------------------------------------------
# Fix 1 — patch is NaN-free for |z| ≤ 1 at any p_range
# ----------------------------------------------------------------------------

class TestFix1PatchNoNaN:
    """Patch dispatch must never produce NaN for |z| < 1, even at p_range
    where the always-evaluated "zero" branch would overflow."""

    @pytest.mark.parametrize("p_range", [50, 200, 500, 1000])
    @pytest.mark.parametrize("s", [2, 3])
    @pytest.mark.parametrize("im_arg", [3.0, 5.0, 8.0, 15.0, 30.0])
    def test_patch_finite_at_LCS_tiny(self, s, im_arg, p_range):
        """``z = e^{2πi (0.3 + i·im_arg)}`` covers ``|z|`` from 1e-9 down
        to ~1e-82.  Pre-v0.3.0, p_range=500 returned NaN for all such z."""
        z = jnp.exp(2j * jnp.pi * (0.3 + 1j * im_arg))
        v = complex(jax_polylog(z, s, p_range, "patch", PVAL))
        assert np.isfinite(v), f"NaN at s={s}, im_arg={im_arg}, p_range={p_range}"


# ----------------------------------------------------------------------------
# Fix 2 — zero branch works at any p_range
# ----------------------------------------------------------------------------

class TestFix2ZeroNoOverflow:
    """The ``"zero"`` branch must accept any ``p_range`` without overflowing
    the Bernoulli table.  Pre-v0.3.0, p_range ≥ ~250 produced NaN."""

    @pytest.mark.parametrize("p_range", [200, 500, 1000])
    def test_zero_branch_finite_at_z_near_1(self, p_range):
        z = jnp.asarray(0.95 + 0.05j)
        v = complex(jax_polylog(z, 3, p_range, "zero", PVAL))
        assert np.isfinite(v)
        # Reference: ζ(3)-truncated series at z=0.95+0.05j
        ref = complex(mp_polylog(3, complex(z)))
        assert _relerr(v, ref) < 1e-13

    def test_zero_branch_matches_mpmath_outside_disk(self):
        """Patch internally routes |z|>1 to the zero branch — pin its accuracy."""
        z = jnp.asarray(1.5 - 0.5j)
        v = complex(jax_polylog(z, 3, 500, "zero", PVAL))
        ref = complex(mp_polylog(3, complex(z)))
        assert _relerr(v, ref) < 1e-13


# ----------------------------------------------------------------------------
# Fix 3 — _Li_over_z(s=1) Taylor fallback cures s=2 high-order cancellation
# ----------------------------------------------------------------------------

class TestFix3Li1OverZStable:
    """d^n Li_2(z)/dz^n at small |z| must hold fp64 precision through n=8.
    Pre-v0.3.0, n ≥ 2 rapidly degenerated to rel-err ~1 because autodiff of
    the literal ``-log1p(-z)/z`` cascades cancelling 1/z^k terms."""

    @pytest.mark.parametrize("z_val", [
        1e-20 + 1e-21j,         # LCS-tiny
        1e-6  + 1e-7j,          # LCS-mid
        0.1   + 0.05j,          # intermediate (just inside the EPS=0.5 cutoff)
    ])
    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    def test_d_n_Li2_at_small_z(self, z_val, n):
        z = jnp.asarray(z_val + 0j)
        f = lambda x: jax_polylog(x, 2, 500, "patch", PVAL)
        v = complex(_nth_holomorphic_deriv(f, n)(z))
        ref = complex(mp_diff(lambda zz: mp_polylog(2, zz), complex(z_val), n))
        # fp64 precision needs ~ |z|^(60-n) · n!/(60-n)! safety margin from
        # the 60-term truncation in :func:`_Li1_over_z_stable`.  For n ≤ 8
        # and |z| < 0.5, this is well under 1e-12.
        assert np.isfinite(v), f"NaN at n={n}, z={z_val}"
        err = _relerr(v, ref)
        assert err < 1e-10, f"n={n} z={z_val}: rel-err {err:.2e}, expected <1e-10"


# ----------------------------------------------------------------------------
# Fix 1 — vmap of patch over mixed LCS + conifold inputs
# ----------------------------------------------------------------------------

class TestPatchVmapMixedInputs:
    """vmapped ``patch`` must produce no NaN when the batch mixes
    LCS-tiny (small ``|z|``) and conifold-approaching (``|z| ≈ 1``)
    elements.  This is the canonical use case the patch dispatch is
    designed for."""

    def test_vmap_pure_LCS(self):
        rng = np.random.default_rng(13)
        zs = np.array([np.exp(rng.uniform(-50, -10) + 1j*rng.uniform(-np.pi, np.pi))
                       for _ in range(20)])
        out = jax_polylog_vmap(jnp.asarray(zs + 0j), 3, 500, "patch", PVAL)
        out_np = np.asarray(out)
        assert np.all(np.isfinite(out_np)), f"NaN at {np.where(~np.isfinite(out_np))}"
        ref = np.array([complex(mp_polylog(3, complex(z))) for z in zs])
        errs = np.array([_relerr(o, r) for o, r in zip(out_np, ref)])
        assert errs.max() < 1e-13

    def test_vmap_pure_conifold(self):
        rng = np.random.default_rng(14)
        zs = np.array([rng.uniform(0.95, 0.999) * np.exp(1j*rng.uniform(-0.1, 0.1))
                       for _ in range(20)])
        out = jax_polylog_vmap(jnp.asarray(zs + 0j), 3, 500, "patch", PVAL)
        out_np = np.asarray(out)
        assert np.all(np.isfinite(out_np))
        ref = np.array([complex(mp_polylog(3, complex(z))) for z in zs])
        errs = np.array([_relerr(o, r) for o, r in zip(out_np, ref)])
        assert errs.max() < 1e-13

    def test_vmap_mixed_LCS_and_conifold(self):
        """The canonical multi-modulus conifold scenario: one batch element
        is at the conifold (z near 1) while others are deep in the LCS limit
        (z very small).  Patch must handle both regimes in a single
        compiled program."""
        rng = np.random.default_rng(15)
        z_lcs = np.array([np.exp(rng.uniform(-40, -5) + 1j*rng.uniform(-np.pi, np.pi))
                          for _ in range(8)])
        z_cf = np.array([rng.uniform(0.95, 0.999) * np.exp(1j*rng.uniform(-0.1, 0.1))
                         for _ in range(4)])
        zs = np.concatenate([z_lcs, z_cf])
        out = jax_polylog_vmap(jnp.asarray(zs + 0j), 3, 500, "patch", PVAL)
        out_np = np.asarray(out)
        assert np.all(np.isfinite(out_np))
        ref = np.array([complex(mp_polylog(3, complex(z))) for z in zs])
        errs = np.array([_relerr(o, r) for o, r in zip(out_np, ref)])
        assert errs.max() < 1e-13


# ----------------------------------------------------------------------------
# vmap of patch under autodiff (LCS+conifold mix, first derivative)
# ----------------------------------------------------------------------------

class TestPatchVmapAutodiff:
    """vmap composed with jvp on patch must also be NaN-free across mixed inputs."""

    def test_vmap_d_dz_mixed(self):
        rng = np.random.default_rng(16)
        z_lcs = np.array([np.exp(rng.uniform(-40, -5) + 1j*rng.uniform(-np.pi, np.pi))
                          for _ in range(8)])
        z_cf = np.array([rng.uniform(0.95, 0.999) * np.exp(1j*rng.uniform(-0.1, 0.1))
                         for _ in range(4)])
        zs = jnp.asarray(np.concatenate([z_lcs, z_cf]) + 0j)
        # vmap of d/dz patch
        d_patch = jax.vmap(lambda x: jax.jvp(
            lambda y: jax_polylog(y, 3, 500, "patch", PVAL),
            (x,), (jnp.asarray(1.0 + 0j),)
        )[1])
        out = d_patch(zs)
        out_np = np.asarray(out)
        assert np.all(np.isfinite(out_np)), \
            f"NaN in vmap-jvp patch at idx {np.where(~np.isfinite(out_np))}"