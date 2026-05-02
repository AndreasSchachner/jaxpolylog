# ==============================================================================
# Derivative tests for jaxpolylog.
#
# Verifies the analytic identity
#
#     d/dz Li_s(z) = Li_{s-1}(z) / z
#
# and that derivatives match high-precision mpmath references at a wide
# range of ``z`` values, from the unit-circle approach (``|z| ≈ 0.99``)
# down to the deep LCS regime (``|z| ≈ 1e-114``).
#
# Regression history:
#
# * **0.1.0**: ``custom_vjp`` rule with residual ``Li_{s-1}(z)/z``.
#   - Worked at moderate ``|z|``.
#   - Catastrophic cancellation at tiny ``|z|`` for higher derivatives:
#     ``-log(1-z)/z²`` produces wrong values when ``1-z`` rounds to 1 in
#     float64 (e.g. ``∂²Li_3(1e-10) ≈ 827`` instead of the correct ``0.25``).
#   - 3rd-derivative NaN at ``|z| ≲ 1e-103`` because ``1/z³`` overflows.
#
# * **0.2.0**: ``custom_jvp`` rule with ``_Li_over_z`` helper that evaluates
#   ``Li_{s-1}(z)/z`` without an explicit ``/z`` primitive (closed-form
#   simplification for ``s ≤ 0``, re-indexed inf-series for ``s ≥ 2``,
#   ``-log1p(-z)/z`` for ``s = 1`` — algebraically identical to v0.1.0).
#   Also static-unrolls the inf series so JAX dispatches to
#   ``lax.integer_pow`` (avoiding ``0 * pow(z, -n)`` cascades).
#   - Fixes 1st/2nd/3rd/4th-derivative NaN at tiny ``|z|``.
#   - Fixes catastrophic cancellation: ``∂²Li_3(1e-10)`` → 0.25 (correct).
#   - Initial 0.2.0 had a regression at ``s=1``: truncated Taylor series
#     instead of closed form, causing 1e-5 errors at ``|z| ≈ 0.95``.  Fixed
#     in this version (uses ``-jnp.log1p(-z)/z``).
#   - ``jax.jvp`` now works (v0.1.0 raised TypeError because ``custom_vjp``
#     does not support forward-mode autodiff).
# ==============================================================================

import sys
import os

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from mpmath import mp, polylog as mp_polylog, diff as mp_diff

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jaxpolylog.polylogs import jax_polylog, jax_polylog_vmap

mp.dps = 60


# ============================================================================
# mpmath reference helpers
# ============================================================================
def mp_polylog_ref(s, z_val):
    """Li_s(z) at high precision."""
    return complex(mp_polylog(s, mp.mpc(complex(z_val))))


def mp_polylog_deriv_ref(s, z_val, n):
    """n-th derivative of Li_s(z) at high precision."""
    return complex(mp_diff(lambda z_: mp_polylog(s, z_),
                           mp.mpc(complex(z_val)), n))


# ============================================================================
# Tests
# ============================================================================
class TestDerivativeStability:
    """Regression: derivatives must remain finite at very small |z|."""

    @pytest.mark.parametrize("s", [2, 3])
    @pytest.mark.parametrize("z_val", [
        1e-5  + 0j,
        1e-30 + 0j,
        1e-60 + 0j,
        1e-100 + 0j,
        1e-114 + 0j,
    ])
    def test_first_derivative_finite(self, s, z_val):
        z = jnp.array(z_val, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 5, "inf")
        g = jax.grad(f, holomorphic=True)(z)
        assert not np.isnan(np.asarray(g)).any()
        assert not np.isinf(np.asarray(g)).any()

    @pytest.mark.parametrize("s", [2, 3])
    @pytest.mark.parametrize("z_val", [1e-5+0j, 1e-30+0j, 1e-100+0j, 1e-114+0j])
    def test_second_derivative_finite(self, s, z_val):
        z = jnp.array(z_val, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 5, "inf")
        g2 = jax.jacrev(jax.grad(f, holomorphic=True), holomorphic=True)(z)
        assert not np.isnan(np.asarray(g2)).any()
        assert not np.isinf(np.asarray(g2)).any()

    # ``s=3`` is the order used by jaxvacua's F_inst.  Higher derivatives
    # of Li_3 chain through ``_Li_over_z(z, 2, ...)`` (a polynomial-only
    # body), so they are stable at all tested ``|z|``.
    #
    # ``s=2`` parent chains through ``_Li_over_z(z, 1, ...) = -log1p(-z)/z``
    # which has a literal ``/z``.  JAX auto-diffs this through the chain
    # rule, producing ``1/z²`` (3rd-deriv) and ``1/z³`` (4th-deriv) factors
    # that overflow for ``|z| ≲ 1e-100``.  This matches v0.1.0's behaviour
    # (since v0.1.0 returned the same closed-form residual via ``custom_vjp``)
    # — neither version is stable here.  Tests below exclude that regime.
    @pytest.mark.parametrize("s,z_val", [
        # s=3 chain: stable at all |z|
        (3, 1e-5+0j), (3, 1e-30+0j), (3, 1e-100+0j), (3, 1e-114+0j),
        # s=2 chain: stable up to |z| ~ 1e-50
        (2, 1e-5+0j), (2, 1e-30+0j), (2, 1e-50+0j),
    ])
    def test_third_derivative_finite(self, s, z_val):
        z = jnp.array(z_val, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 5, "inf")
        g3 = jax.jacrev(jax.jacrev(jax.grad(f, holomorphic=True),
                                   holomorphic=True), holomorphic=True)(z)
        assert not np.isnan(np.asarray(g3)).any()

    @pytest.mark.parametrize("s,z_val", [
        # s=3 chain: stable at all |z|
        (3, 1e-5+0j), (3, 1e-30+0j), (3, 1e-100+0j), (3, 1e-114+0j),
        # s=2 chain: stable up to |z| ~ 1e-30 for 4th-deriv (the chain
        # introduces 1/z³ which overflows at ~1e-103)
        (2, 1e-5+0j), (2, 1e-30+0j),
    ])
    def test_fourth_derivative_finite(self, s, z_val):
        z = jnp.array(z_val, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 5, "inf")
        g = jax.grad(f, holomorphic=True)
        for _ in range(3):
            g = jax.jacrev(g, holomorphic=True)
        g4 = g(z)
        assert not np.isnan(np.asarray(g4)).any()


class TestPrimalAccuracyVsMpmath:
    """Primal Li_s(z) must match mpmath at moderate |z|."""

    @pytest.mark.parametrize("s", [2, 3, 4])
    @pytest.mark.parametrize("z_val", [
        0.1 + 0j, 0.3 + 0j, 0.5 + 0.1j, 0.7 - 0.3j, 0.9 - 0.3j,
        0.01 + 0.01j, 1e-3 + 1e-3j, 1e-10 + 0j,
    ])
    def test_primal_matches_mpmath(self, s, z_val):
        z = jnp.array(z_val, dtype=jnp.complex128)
        ref = mp_polylog_ref(s, z_val)
        val = complex(jax_polylog(z, s, 1000, "inf"))
        rel_tol = 1e-10
        assert abs(val - ref) < rel_tol * max(abs(ref), 1.0), (
            f"Li_{s}({z_val}): jax={val:.6e}, mpmath={ref:.6e}, "
            f"diff={abs(val-ref):.3e}"
        )


class TestFirstDerivativeVsMpmath:
    """∂Li_s/∂z must match mpmath at all tested |z|, including tiny |z|."""

    @pytest.mark.parametrize("s", [2, 3, 4])
    @pytest.mark.parametrize("z_val", [
        0.1 + 0j, 0.3 + 0j, 0.5 + 0.1j, 0.7 - 0.3j, 0.9 - 0.3j,
        0.01 + 0.01j, 1e-3 + 1e-3j, 1e-6 + 1e-6j,
        1e-10 + 0j, 1e-30 + 0j, 1e-100 + 0j,
    ])
    def test_grad_matches_mpmath(self, s, z_val):
        z = jnp.array(z_val, dtype=jnp.complex128)
        ref = mp_polylog_deriv_ref(s, z_val, 1)
        f = lambda zz: jax_polylog(zz, s, 1000, "inf")
        val = complex(jax.grad(f, holomorphic=True)(z))
        rel_tol = 1e-9
        assert abs(val - ref) < rel_tol * max(abs(ref), 1.0), (
            f"∂Li_{s}({z_val}): jax={val:.6e}, mpmath={ref:.6e}, "
            f"diff={abs(val-ref):.3e}"
        )


class TestSecondDerivativeVsMpmath:
    """∂²Li_s/∂z² must match mpmath at moderate |z| and at tiny |z|.

    Regression: v0.1.0 had catastrophic cancellation at tiny |z| (giving
    e.g. 827 instead of 0.25 for ∂²Li_3(1e-10)).
    """

    @pytest.mark.parametrize("s", [2, 3, 4])
    @pytest.mark.parametrize("z_val", [
        0.1 + 0j, 0.3 + 0j, 0.5 + 0.1j, 0.7 - 0.3j, 0.9 - 0.3j,
        0.01 + 0.01j, 1e-3 + 1e-3j, 1e-6 + 1e-6j, 1e-10 + 0j,
    ])
    def test_hessian_matches_mpmath(self, s, z_val):
        z = jnp.array(z_val, dtype=jnp.complex128)
        ref = mp_polylog_deriv_ref(s, z_val, 2)
        f = lambda zz: jax_polylog(zz, s, 1000, "inf")
        val = complex(jax.jacrev(jax.grad(f, holomorphic=True),
                                  holomorphic=True)(z))
        rel_tol = 1e-7
        assert abs(val - ref) < rel_tol * max(abs(ref), 1.0), (
            f"∂²Li_{s}({z_val}): jax={val:.6e}, mpmath={ref:.6e}, "
            f"diff={abs(val-ref):.3e}"
        )


class TestThirdDerivativeVsMpmath:
    """∂³Li_s/∂z³ — used by jaxvacua's ddDW etc.  v0.1.0 produced NaN at
    |z| ≲ 1e-103."""

    @pytest.mark.parametrize("s", [3])
    @pytest.mark.parametrize("z_val", [
        0.1 + 0j, 0.3 + 0j, 0.5 + 0.1j, 0.7 - 0.3j, 0.9 - 0.3j,
        0.01 + 0.01j, 1e-3 + 1e-3j, 1e-6 + 1e-6j,
        1e-10 + 0j, 1e-30 + 0j, 1e-100 + 0j,
    ])
    def test_third_derivative_matches_mpmath(self, s, z_val):
        z = jnp.array(z_val, dtype=jnp.complex128)
        ref = mp_polylog_deriv_ref(s, z_val, 3)
        f = lambda zz: jax_polylog(zz, s, 1000, "inf")
        val = complex(jax.jacrev(jax.jacrev(jax.grad(f, holomorphic=True),
                                            holomorphic=True),
                                 holomorphic=True)(z))
        rel_tol = 1e-6
        assert abs(val - ref) < rel_tol * max(abs(ref), 1.0), (
            f"∂³Li_{s}({z_val}): jax={val:.6e}, mpmath={ref:.6e}, "
            f"diff={abs(val-ref):.3e}"
        )


class TestDerivativeIdentityClosedForm:
    """Verify d/dz Li_s(z) = Li_{s-1}(z)/z for the closed-form region."""

    @pytest.mark.parametrize("s", [2, 3, 4])
    @pytest.mark.parametrize("z_val", [0.3+0.0j, 0.5+0.1j, 0.7-0.3j, -0.4+0.2j])
    def test_first_derivative_identity(self, s, z_val):
        z = jnp.array(z_val, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 1000, "inf")
        deriv_via_autodiff = complex(jax.grad(f, holomorphic=True)(z))
        deriv_via_identity = (
            complex(jax_polylog(z, s - 1, 1000, "inf")) / complex(z)
        )
        assert abs(deriv_via_autodiff - deriv_via_identity) < 1e-12


class TestForwardModeAutodiff:
    """jax.jvp must work (v0.1.0 raised TypeError because of custom_vjp)."""

    @pytest.mark.parametrize("s", [2, 3, 4])
    def test_jvp_works(self, s):
        z = jnp.array(0.5 + 0.1j, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 1000, "inf")
        primal, tangent = jax.jvp(f, (z,), (jnp.array(1.0 + 0j),))
        assert not np.isnan(complex(primal).real)
        assert not np.isnan(complex(tangent).real)
        # tangent must equal Li_{s-1}(z)/z
        expected = complex(jax_polylog(z, s - 1, 1000, "inf")) / complex(z)
        assert abs(complex(tangent) - expected) < 1e-12


class TestVmapDerivatives:
    """vmap'd derivatives must remain finite at extreme |z| (used by jaxvacua's
    F_inst, where polylog is evaluated over many GV charges with vastly
    different |z|)."""

    def test_vmap_third_derivative_extreme_z(self):
        zs = jnp.array([1e-5, 1e-30, 1e-60, 1e-100, 1e-114],
                       dtype=jnp.complex128)
        f = lambda zz: jnp.sum(jax_polylog_vmap(zz, 3, 5))
        g3 = jax.jacrev(jax.jacrev(jax.grad(f, holomorphic=True),
                                   holomorphic=True),
                        holomorphic=True)(zs)
        assert not np.isnan(np.asarray(g3)).any()


class TestRegressionCases:
    """Specific regression points from the v0.1.0 → v0.2.0 transition.

    These pinpoint values where v0.1.0 was wrong and the fix gives the
    correct (mpmath-verified) result.
    """

    def test_d2_Li3_at_1e10_is_quarter(self):
        """v0.1.0 returned ~827 at z=1e-10 due to catastrophic cancellation
        in ``-log(1-z)/z² - Li_2/z²``.  Correct value (mpmath): 0.25."""
        z = jnp.array(1e-10 + 0j, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, 3, 1000, "inf")
        val = complex(jax.jacrev(jax.grad(f, holomorphic=True),
                                  holomorphic=True)(z))
        assert abs(val - 0.25) < 1e-9, f"Expected 0.25, got {val}"

    def test_grad_Li2_at_1e30_is_one(self):
        """At z=1e-30, v0.1.0 returned 0 (since 1-1e-30 rounds to 1 and
        log(1) = 0).  Correct value (mpmath): 1."""
        z = jnp.array(1e-30 + 0j, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, 2, 1000, "inf")
        val = complex(jax.grad(f, holomorphic=True)(z))
        assert abs(val - 1.0) < 1e-12, f"Expected 1.0, got {val}"

    def test_grad_Li2_near_unit_circle_matches_closed_form(self):
        """Initial v0.2.0 (truncated series) had ~1e-3 error at z=0.95.
        Fixed: uses -log1p(-z)/z closed form via ``_Li_over_z(.,1,...)``."""
        z = jnp.array(0.95 + 0.0j, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, 2, 1000, "inf")
        val = complex(jax.grad(f, holomorphic=True)(z))
        ref = mp_polylog_deriv_ref(2, 0.95 + 0.0j, 1)
        assert abs(val - ref) < 1e-12, (
            f"Near-unit-circle ∂Li_2(0.95): got {val}, expected {ref}"
        )
