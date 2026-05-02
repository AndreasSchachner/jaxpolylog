# ==============================================================================
# Derivative tests for jaxpolylog.
#
# Verifies the analytic identity
#
#     d/dz Li_s(z) = Li_{s-1}(z) / z
#
# and that higher-order derivatives (∂², ∂³, ∂⁴) are returned as finite,
# correct values down to ``|z| ~ 1e-114`` — the regime hit by Calabi–Yau
# period integrals at large complex-structure (LCS).
#
# Regression tests for two bugs fixed in v0.2.0:
#   - 3rd derivative produced NaN at ``|z| < 1e-103`` because the previous
#     ``custom_vjp`` rule's residual ``Li_{s-1}(z)/z`` led JAX to cascade
#     ``1/z``, ``1/z²``, ``1/z³`` as standalone primitives during higher-
#     order autodiff; ``1/z³`` overflowed float64 (>1e+308).  Fixed by
#     switching to ``custom_jvp`` and computing ``Li_{s-1}(z)/z`` via
#     ``_Li_over_z`` (algebraically identical, no division by ``z``).
#   - 4th derivative produced NaN at ``|z| < 1e-103`` because the inf-series
#     was evaluated as ``z**polylog_range`` over a *traced* arange; JAX's
#     general ``pow`` op derivative ``n * pow(z, n-1)`` introduced
#     ``0 * pow(z, -n)`` factors for the ``k=1`` term that overflowed.
#     Fixed by hand-extracting the ``k=1`` constant and unrolling the rest
#     as a Python loop with static integer exponents (``lax.integer_pow``).
#
# In addition, ``jax.jvp`` now works (previously raised
# ``TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp
# function``) because the new rule is a ``custom_jvp`` (JAX auto-derives
# the VJP from the JVP via transposition).
# ==============================================================================

import sys
import os

import pytest
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jaxpolylog.polylogs import jax_polylog, jax_polylog_vmap


# ============================================================================
# Tests
# ============================================================================
class TestDerivativeStability:
    """Regression: derivatives of Li_s(z) must be finite for tiny |z|."""

    @pytest.mark.parametrize("s", [2, 3])
    @pytest.mark.parametrize("z_val", [
        1e-5  + 0j,
        1e-30 + 0j,
        1e-60 + 0j,
        1e-100 + 0j,
        1e-114 + 0j,
    ])
    def test_first_derivative_finite(self, s, z_val):
        """∂Li_s/∂z must be finite at all tested ``z``."""
        z = jnp.array(z_val, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 5, "inf")
        g = jax.grad(f, holomorphic=True)(z)
        assert not np.isnan(np.asarray(g)).any(), f"∂Li_{s}/∂z is NaN at z={z_val}"
        assert not np.isinf(np.asarray(g)).any(), f"∂Li_{s}/∂z is inf at z={z_val}"

    @pytest.mark.parametrize("s", [2, 3])
    @pytest.mark.parametrize("z_val", [1e-5+0j, 1e-30+0j, 1e-100+0j, 1e-114+0j])
    def test_second_derivative_finite(self, s, z_val):
        """∂²Li_s/∂z² must be finite (regression for old `1/z²` cascade)."""
        z = jnp.array(z_val, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 5, "inf")
        g2 = jax.jacrev(jax.grad(f, holomorphic=True), holomorphic=True)(z)
        assert not np.isnan(np.asarray(g2)).any()
        assert not np.isinf(np.asarray(g2)).any()

    @pytest.mark.parametrize("s", [2, 3])
    @pytest.mark.parametrize("z_val", [1e-5+0j, 1e-30+0j, 1e-100+0j, 1e-114+0j])
    def test_third_derivative_finite(self, s, z_val):
        """∂³Li_s/∂z³ must be finite (regression for old `1/z³` overflow)."""
        z = jnp.array(z_val, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 5, "inf")
        g3 = jax.jacrev(jax.jacrev(jax.grad(f, holomorphic=True),
                                   holomorphic=True), holomorphic=True)(z)
        assert not np.isnan(np.asarray(g3)).any(), f"∂³Li_{s}/∂z³ is NaN at z={z_val}"

    @pytest.mark.parametrize("s", [2, 3])
    @pytest.mark.parametrize("z_val", [1e-5+0j, 1e-30+0j, 1e-100+0j, 1e-114+0j])
    def test_fourth_derivative_finite(self, s, z_val):
        """∂⁴Li_s/∂z⁴ must be finite (regression for `0 * pow(z, -n)`)."""
        z = jnp.array(z_val, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 5, "inf")
        g = jax.grad(f, holomorphic=True)
        for _ in range(3):
            g = jax.jacrev(g, holomorphic=True)
        g4 = g(z)
        assert not np.isnan(np.asarray(g4)).any(), f"∂⁴Li_{s}/∂z⁴ is NaN at z={z_val}"


class TestDerivativeIdentity:
    """Verify the analytic identity d/dz Li_s(z) = Li_{s-1}(z)/z."""

    @pytest.mark.parametrize("s", [2, 3, 4])
    @pytest.mark.parametrize("z_val", [0.3+0.0j, 0.5+0.1j, 0.7-0.3j, -0.4+0.2j])
    def test_first_derivative_identity(self, s, z_val):
        """∂Li_s/∂z = Li_{s-1}(z)/z (basic identity check)."""
        z = jnp.array(z_val, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 1000, "inf")
        deriv_via_autodiff = complex(jax.grad(f, holomorphic=True)(z))
        # Reference: Li_{s-1}(z)/z computed independently
        deriv_via_identity = complex(jax_polylog(z, s - 1, 1000, "inf")) / complex(z)
        assert abs(deriv_via_autodiff - deriv_via_identity) < 1e-12, (
            f"d/dz Li_{s}({z_val}) = {deriv_via_autodiff:.6e}, "
            f"Li_{s-1}/z = {deriv_via_identity:.6e}"
        )


class TestForwardModeAutodiff:
    """jax.jvp must work (regression: custom_vjp blocked forward-mode AD)."""

    @pytest.mark.parametrize("s", [2, 3, 4])
    def test_jvp_works(self, s):
        z = jnp.array(0.5 + 0.1j, dtype=jnp.complex128)
        f = lambda zz: jax_polylog(zz, s, 1000, "inf")
        # Should not raise TypeError
        primal, tangent = jax.jvp(f, (z,), (jnp.array(1.0 + 0j),))
        assert not np.isnan(complex(primal).real)
        assert not np.isnan(complex(tangent).real)
        # tangent must equal Li_{s-1}(z)/z
        expected = complex(jax_polylog(z, s - 1, 1000, "inf")) / complex(z)
        assert abs(complex(tangent) - expected) < 1e-12


class TestVmapDerivatives:
    """vmap'd derivatives must also be finite at extreme |z|."""

    def test_vmap_grad_extreme_z(self):
        """Test the vmap'd version (used by jaxvacua's F_inst over GV charges)."""
        # Mimic F_inst: many polylog evaluations at exp(2πi · q · moduli) for
        # various charges q.  At LCS, |z| can range from ~1e-22 down to ~1e-114.
        zs = jnp.array([1e-5, 1e-30, 1e-60, 1e-100, 1e-114], dtype=jnp.complex128)
        f = lambda zz: jnp.sum(jax_polylog_vmap(zz, 3, 5))
        g3 = jax.jacrev(jax.jacrev(jax.grad(f, holomorphic=True),
                                   holomorphic=True), holomorphic=True)(zs)
        assert not np.isnan(np.asarray(g3)).any()
