# ==============================================================================
# This code is written by Andreas Schachner. Without the author's permission, this code must not be shared with anyone else or used for any other projects than those involving the author directly.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu or at a.schachner@lmu.de.
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file holds functions for polylogarithms using JAX.
# ------------------------------------------------------------------------------


# Important standard libraries
import os, sys, warnings
import math
from fractions import Fraction
from functools import partial, lru_cache

# Important JAX libraries
import jax
from jax import custom_jvp
from jax import jit, vmap, config
import jax.numpy as jnp
from jax import Array
from numpy.typing import ArrayLike

# Enable 64 bit precision
config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Stable coefficient utilities for the ``"zero"`` series expansion.
# ---------------------------------------------------------------------------
# The ``"zero"`` branch evaluates  Li_s(z) = Σ_k ζ(s-k) / k! · μ^k  with
# μ = log z, which converges for |μ| < 2π.  The kth coefficient is
# ζ(s-k)/k!.  The old implementation built this via
#   ``Bs = jax.scipy.special.bernoulli(p_range); zeta_neg = -Bs/k``
# which overflows fp64 for ``p_range ≳ 250`` because individual
# Bernoulli numbers grow factorially even though the coefficient
# B_k / k! we actually need decays geometrically.  The helpers below
# compute the coefficients without ever materialising large ``B_k`` as
# floats, using exact :class:`fractions.Fraction` Bernoulli numbers and
# (where needed) the identity
#   ζ(2m) = |B_{2m}| · (2π)^{2m} / (2 · (2m)!).
# All computation happens at module-trace time and is cached.


@lru_cache(maxsize=4)
def _bernoulli_fractions_up_to(N: int) -> tuple:
    r"""Return ``(B_0, B_1, ..., B_N)`` as :class:`fractions.Fraction`.

    Akiyama–Tanigawa recurrence, exact rationals.  Used by
    :func:`_zero_branch_coeffs` and :func:`_zeta_pos_int` for stable
    coefficient construction.  Cached because computation is O(N²) in
    bignum operations.
    """
    a = [Fraction(0)] * (N + 1)
    B = [Fraction(0)] * (N + 1)
    for m in range(N + 1):
        a[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            a[j - 1] = j * (a[j - 1] - a[j])
        B[m] = a[0]
    return tuple(B)


@lru_cache(maxsize=64)
def _zeta_pos_int(n: int) -> float:
    r"""Riemann ``ζ(n)`` for integer ``n ≥ 2``, fp64-precise.

    Even ``n``: closed form ``ζ(2m) = |B_{2m}| (2π)^{2m} / (2 · (2m)!)`` from
    the exact Bernoulli table.  Odd ``n``: direct summation plus a five-term
    Euler–Maclaurin tail (more than enough for fp64 from K=60 onward).
    """
    if n < 2:
        raise ValueError(f"_zeta_pos_int requires n >= 2, got {n}")
    if n % 2 == 0:
        B = _bernoulli_fractions_up_to(n)
        abs_bn = B[n] if B[n] > 0 else -B[n]
        ratio = abs_bn / (Fraction(2) * Fraction(math.factorial(n)))
        return float(ratio) * (2.0 * math.pi) ** n
    K = 60
    Kf = float(K)
    s = sum(1.0 / k ** n for k in range(1, K))
    s += 1.0 / ((n - 1) * Kf ** (n - 1))
    s += 0.5 / Kf ** n
    s += n / (12.0 * Kf ** (n + 1))
    s -= n * (n + 1) * (n + 2) / (720.0 * Kf ** (n + 3))
    s += n * (n + 1) * (n + 2) * (n + 3) * (n + 4) / (30240.0 * Kf ** (n + 5))
    return s


# Hard cap on the number of terms used in the "zero" Laurent series.  The
# series converges geometrically with ratio |μ|/(2π); 200 terms gives
# truncation error ≤ 0.5^200 ≈ 10^{-60} at the patch threshold |μ|/2π ≈ pval,
# so any larger ``p_range`` adds no precision but slows the precomputation.
_ZERO_BRANCH_P_MAX = 200


@lru_cache(maxsize=64)
def _zero_branch_coeffs(s: int, P: int) -> tuple:
    r"""Precompute ``c[k] = ζ(s-k) / k!`` for ``k = 0..P-1``.

    Branch table:

      * ``arg = s-k ≥ 2``:  ``ζ(arg)`` via :func:`_zeta_pos_int`.
      * ``arg = 1`` (``k = s-1``):  coefficient set to 0 (cancels the
        explicit ``log(-μ)`` term in :func:`jax_polylog`).
      * ``arg = 0`` (``k = s``):    ``ζ(0) = -1/2``.
      * ``arg < 0``, ``n = 1 - arg``:
          - ``n`` odd ≥ 3:  ``B_n = 0`` → coefficient = 0.
          - ``n = 2m`` even:  coefficient ``= -B_{2m}/(2m · k!)`` as an
            exact :class:`Fraction`, then floated.  Never materialises
            ``B_{2m}`` or ``k!`` separately.
    """
    coeffs = [0.0] * P
    Bf = _bernoulli_fractions_up_to(max(2, P)) if P > 1 else None
    for k in range(P):
        if k == s - 1:
            continue
        arg = s - k
        if arg >= 2:
            coeffs[k] = _zeta_pos_int(arg) / math.factorial(k)
        elif arg == 1:
            continue
        elif arg == 0:
            coeffs[k] = -0.5 / math.factorial(k)
        else:
            n = 1 - arg
            if n == 1 or (n % 2 == 1):
                continue
            frac = -Bf[n] / (Fraction(n) * Fraction(math.factorial(k)))
            coeffs[k] = float(frac)
    return tuple(coeffs)


def _Li1_over_z_stable(z):
    r"""``Li_1(z)/z = -log1p(-z)/z`` evaluated stably under nested autodiff.

    The closed form ``-log1p(-z)/z`` is correct numerically for any ``|z|``,
    but JAX autodiff cascades produce catastrophic cancellation at small
    ``|z|`` from the second derivative onward: ``d/dz[-log1p(-z)/z]`` is
    algebraically ``1/(z(1-z)) + log1p(-z)/z²``, two ``O(1/z)`` terms that
    cancel mathematically but lose all bits in fp64 for ``|z| ≲ 10⁻²``.

    Cure: at ``|z| < 0.5`` use the convergent power series

    .. math::
        \frac{\mathrm{Li}_1(z)}{z} \;=\; \sum_{k=0}^{\infty} \frac{z^k}{k+1}
        \;=\; 1 + \frac{z}{2} + \frac{z^2}{3} + \cdots,

    a polynomial in ``z`` whose JAX autodiff is exact at all orders.  60 terms
    give truncation error ``|z|^{60} ≤ 0.5^{60} ≈ 10⁻¹⁸`` at the cutoff.  The
    double-where ensures the inactive branch sees a safe argument so that
    ``0 · NaN = NaN`` poisoning cannot reach the result.
    """
    EPS = 0.5
    use_series = jnp.abs(z) < EPS
    SAFE_CLOSED = jnp.asarray(0.7 + 0.0j, dtype=z.dtype)
    SAFE_SERIES = jnp.asarray(0.3 + 0.0j, dtype=z.dtype)
    z_series = jnp.where(use_series, z, SAFE_SERIES)
    z_closed = jnp.where(use_series, SAFE_CLOSED, z)

    series_val = jnp.ones_like(z_series)
    zk = z_series
    for k in range(2, 62):
        series_val = series_val + zk / float(k)
        zk = zk * z_series
    closed_val = -jnp.log1p(-z_closed) / z_closed
    return jnp.where(use_series, series_val, closed_val)


def _compute_pval_optimal() -> float:
    r"""
    Find the optimal transition parameter ``t*`` for the ``"patch"`` method.

    The ``"patch"`` method switches between two series expansions of Li_s(z):

    * **"inf"** series: ``Li_s(z) = Σ z^k / k^s``, convergent for ``|z| < 1``.
      After ``N`` terms the truncation error scales as ``|z|^N``.
    * **"zero"** expansion: Laurent series in ``μ = log z`` around ``μ = 0``
      (``z = 1``), convergent for ``|μ| < 2π``.  After ``N`` terms the error
      scales as ``(|μ|/(2π))^N = t^N`` where ``t = |μ|/(2π)``.

    For real positive ``z < 1`` both errors are equal when

    .. math::
        |z|^N = t^N \;\Longrightarrow\; |z| = t \;\Longrightarrow\;
        e^{-|μ|} = \frac{|μ|}{2π} \;\Longrightarrow\; e^{-2πt} = t \,.

    The unique positive solution of ``e^{-2πt} = t`` is computed here by
    bisection and stored as the module constant :data:`_PVAL_OPTIMAL` ≈ 0.2322.
    This fixed point is independent of ``N`` (the ``p_range`` parameter), so
    the optimal crossover does not change with the number of series terms.

    Returns:
        float: Optimal transition parameter ``t*`` ≈ 0.2322.
    """
    import numpy as _np
    # Bisect on f(t) = e^{-2πt} - t.  f(0.05) > 0, f(0.50) < 0.
    lo, hi = 0.05, 0.50
    for _ in range(80):           # 80 iterations → sub-ULP accuracy
        mid = 0.5 * (lo + hi)
        if _np.exp(-2.0 * _np.pi * mid) > mid:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Optimal patch transition: t* = |log z| / (2π) at which the truncation errors
# of the "inf" series and the "zero" expansion are equal.  Precomputed once at
# import time by solving e^{-2πt} = t via bisection (~0.2322).
_PVAL_OPTIMAL: float = _compute_pval_optimal()

@partial(jit, static_argnums = (2,))
def intgrand(z: complex, t: complex, s: int) -> complex:
    r"""
    **Description:**
    Integrand for the integral representation of the polylogarithm function.
    
    Args:
        z (complex): The input value(s) at which to evaluate the integrand. Can be a scalar or an array.
        t (complex): The integration variable.
        s (int): The order of the polylogarithm. Must be an integer.  
        
    Returns:
        complex: The computed integrand values.
    
    """
    return jnp.log(t)**(s-1)/(1-z*t) # type: ignore


@partial(custom_jvp, nondiff_argnums=(1,2,3,4,))
@partial(jit, static_argnums=(1,2,3,4,))
def jax_polylog(z: complex, s: int, p_range: int, approx: str, pval: float = _PVAL_OPTIMAL) -> complex: # type: ignore
    r"""
    **Description:**
    This function computes the polylogarithm of order `s` at point `z` using JAX. It supports automatic differentiation and is optimized for performance. The function is defined for integer values of `s` and can handle both real and complex inputs for `z`. 
    
    **Mathematical Definition:**
    The polylogarithm function is defined as:
    
    .. math::
        \text{Li}_s(z) = \sum_{k=1}^{\infty} \frac{z^k}{k^s}
    
    for |z| < 1 and can be analytically continued to other values of `z`.
    For integer `s`, the function can be expressed in terms of elementary functions for specific values of `s`: 
    - For `s = 1`: `Li_1(z) = -ln(1 - z)`
    - For `s = 0`: `Li_0(z) = z / (1 - z)`
    - For `s = -1`: `Li_{-1}(z) = z / (1 - z)^2`
    - For `s = -2`: `Li_{-2}(z) = z(1 + z) / (1 - z)^3`
    - For `s = -3`: `Li_{-3}(z) = z(1 + 4z + z^2) / (1 - z)^4`
    - For `s = -4`: `Li_{-4}(z) = z(1 + 11z + 11z^2 + z^3) / (1 - z)^5`
    - For `s = -5`: `Li_{-5}(z) = z(1 + 26z + 66z^2 + 26z^3 + z^4) / (1 - z)^6`
    - For `s = -6`: `Li_{-6}(z) = z(1 + 57z + 302z^2 + 302z^3 + 57z^4 + z^5) / (1 - z)^7`
    - For `s = -7`: `Li_{-7}(z) = z(1 + 120z + 1191z^2 + 2416z^3 + 1191z^4 + 120z^5 + z^6) / (1 - z)^8`
    - For `s = -8`: `Li_{-8}(z) = z(1 + 247z + 4293z^2 + 15619z^3 + 15619z^4 + 4293z^5 + 247z^6 + z^7) / (1 - z)^9`
    - For `s = -9`: `Li_{-9}(z) = z(1 + 502z + 14608z^2 + 88234z^3 + 156190z^4 + 88234z^5 + 14608z^6 + 502z^7 + z^8) / (1 - z)^10`
    - For other integer values of `s`, the function is computed using the series definition.
    
    Args:
        z (complex): The input value(s) at which to evaluate the polylogarithm. Can be a scalar or an array.
        s (int): The order of the polylogarithm. Must be an integer.
        p_range (int): The number of terms to include in the series expansion for non-predefined `s` values. Higher values increase accuracy but also computation time.
        approx (str): The approximation method to use. Must be one of ``"inf"``, ``"integral"``, ``"zero"``, or ``"patch"``.
        pval (float, optional): Transition parameter for the ``"patch"`` method.
            Points with ``t = |log z| / (2π) < pval`` use the ``"zero"`` expansion;
            points with ``t ≥ pval`` *and* ``|z| < 1`` use the ``"inf"`` series.
            Points with ``|z| ≥ 1`` always use ``"zero"`` (the ``"inf"`` series
            diverges there).  The default is :data:`_PVAL_OPTIMAL` ≈ 0.2322,
            the fixed point of ``e^{-2πt} = t`` that equates both truncation
            errors.  Ignored for ``approx`` values other than ``"patch"``.
        
    Returns:
        complex: The computed polylogarithm values at the input `z`.
        
    Raises:
        ValueError: If `s` is not an integer or if `approx` is not one of the specified methods.
        
    **Example Usage:**
    ```python
    import jax.numpy as jnp
    from jaxvacua.polylogs import jax_polylog as polylog
    z = jnp.array([0.5, 0.9, 1.0])
    s = 2
    result = polylog(z, s, p_range=1000)
    print(result)
    ```
    
    """
    # Check if s is an integer
    if not isinstance(s, int):
        raise ValueError("The order 's' must be an integer.")
    
    if approx not in ["inf","integral","zero","patch"]:
        raise ValueError("The approximation method must be one of 'inf', 'integral', 'zero', or 'patch'.")
    
    # Handle special cases for specific integer values of s
    if s==1:
        return -jnp.log(1-z) # type: ignore
    elif s==0:
        return z/(1-z)
    elif s==-1:
        return z/(1-z)**2
    elif s==-2:
        return ((z*(1 + z))/(1 - z)**3)
    elif s==-3:
        return (z*(1 + 4*z + z**2))/(-1 + z)**4
    elif s==-4:
        return ((z*(1 + 11*z + 11*z**2 + z**3))/(1 - z)**5)
    elif s==-5:
        return (z*(1 + 26*z + 66*z**2 + 26*z**3 + z**4))/(-1 + z)**6
    elif s==-6:
        return ((z*(1 + 57*z + 302*z**2 + 302*z**3 + 57*z**4 + z**5))/(1 - z)**7)
    elif s==-7:
        return (z*(1 + 120*z + 1191*z**2 + 2416*z**3 + 1191*z**4 + 120*z**5 + z**6))/(-1 + z)**8
    elif s==-8:
        return ((z*(1 + 247*z + 4293*z**2 + 15619*z**3 + 15619*z**4 + 4293*z**5 + 247*z**6 + z**7))/(1 - z)**9)
    elif s==-9:
        return (z*(1 + 502*z + 14608*z**2 + 88234*z**3 + 156190*z**4 + 88234*z**5 + 14608*z**6 + 502*z**7 + z**8))/(-1 + z)**10
    else:
        if approx=="inf":
            # Use series definition around z=infty
            polylog_range = jnp.arange(1,p_range)
            return jnp.sum(z**polylog_range/polylog_range**s) # type: ignore
        elif approx=="integral":
            # Use integral representation
            polylog_range = jnp.linspace(0+1e-20,1.,p_range)
            return z*(-1)**(s-1)/jax.scipy.special.gamma(s)*jax.scipy.integrate.trapezoid(intgrand(z,polylog_range,s),x=polylog_range)
        elif approx=="zero":
            # Series expansion around z=1: Li_s(z) = term1 + Σ_k ζ(s-k)/k! · μ^k,
            # convergent for |μ| < 2π with μ = log z.
            #
            # The coefficient table c[k] = ζ(s-k)/k! is precomputed once per
            # (s, P) by :func:`_zero_branch_coeffs` using exact Bernoulli
            # numbers (Fraction arithmetic).  This replaces the old
            # ``jax.scipy.special.bernoulli(p_range)`` call, which overflows
            # fp64 for ``p_range ≳ 250`` because individual B_n grow
            # factorially even though the coefficient B_n/n! decays
            # geometrically.  See :func:`_zero_branch_coeffs` for details.
            mu = jnp.log(z)
            Hs = jnp.sum(1.0 / jnp.arange(1, s))
            term1 = mu ** (s - 1) / jax.scipy.special.gamma(s) * (Hs - jnp.log(-mu))

            P = min(p_range, _ZERO_BRANCH_P_MAX)
            coeffs = jnp.asarray(_zero_branch_coeffs(s, P))
            powers = mu ** jnp.arange(P)
            term2 = jnp.sum(coeffs * powers)
            return term1 + term2  # type: ignore
        elif approx=="patch":
            # Double-where dispatch: both branches must be traced for vmap,
            # but each branch is *evaluated* on a safe argument when inactive.
            # This avoids IEEE ``0 · NaN = NaN`` poisoning when one branch
            # overflows (the "zero" branch at small |z|, where μ^k → huge)
            # or diverges (the "inf" branch at |z| ≥ 1).
            #
            #   |z| < 1  AND  t = |log z|/(2π) ≥ pval   →   "inf"  branch
            #   otherwise                                 →   "zero" branch
            #
            # The safe substitutes (|z|=0.5 for the inf branch, |z|=0.9 for
            # the zero branch) sit comfortably inside each branch's
            # convergence basin, so the inactive branch produces a finite
            # (irrelevant) number that ``jnp.where`` then discards.
            mu = jnp.log(z)
            t  = jnp.abs(mu) / (2.0 * jnp.pi)
            use_inf = (jnp.abs(z) < 1.0) & (t >= pval)

            SAFE_INF  = jnp.asarray(0.5 + 0.0j, dtype=z.dtype)
            SAFE_ZERO = jnp.asarray(0.9 + 0.0j, dtype=z.dtype)
            z_for_inf  = jnp.where(use_inf, z, SAFE_INF)
            z_for_zero = jnp.where(use_inf, SAFE_ZERO, z)

            Lis_inf  = jax_polylog(z_for_inf,  s, p_range, "inf",  pval)
            Lis_zero = jax_polylog(z_for_zero, s, p_range, "zero", pval)
            return jnp.where(use_inf, Lis_inf, Lis_zero)

def _Li_over_z(z: complex, s: int, p_range: int, approx: str, pval: float) -> complex:
    r"""
    **Description:**
    Compute :math:`\mathrm{Li}_s(z)/z` via a numerically stable evaluation that
    avoids dividing by ``z``.

    This quantity is the analytic derivative
    :math:`\mathrm{d}/\mathrm{d}z\,\mathrm{Li}_{s+1}(z) = \mathrm{Li}_s(z)/z`
    used by the JVP rule of :func:`jax_polylog`.  It is mathematically
    identical to ``jax_polylog(z, s, ...)/z`` but evaluated as either a
    closed-form polynomial-rational expression (for ``s ≤ 1``) or a
    re-indexed power series (for ``s ≥ 2`` with ``approx="inf"``).
    Avoiding the explicit division by ``z`` prevents :math:`1/z^n` factors
    from cascading through higher-order autodiff and overflowing float64
    when :math:`|z|` is very small (as happens in LCS-regime period
    integrals where :math:`|z| = e^{-2\pi q\cdot\mathrm{Im}(t)}` may be
    :math:`\lesssim 10^{-100}`).

    Args:
        z (complex): The input value(s).
        s (int): The order parameter (one less than the order of the parent
            polylogarithm being differentiated).
        p_range (int): Number of terms in the series expansion.
        approx (str): Approximation method.  See :func:`jax_polylog`.
        pval (float): Transition parameter for the ``"patch"`` method.

    Returns:
        complex: The value :math:`\mathrm{Li}_s(z)/z` evaluated stably.
    """
    # Closed-form simplifications.  Each entry is the closed-form Li_s(z)
    # divided by z and algebraically simplified — no division by z remains
    # for s <= 0 (the polynomial-rational forms manifestly cancel z).
    if s == 1:
        # Li_1(z)/z = -log(1-z)/z.  Use the closed form (with ``log1p`` for
        # accuracy when ``|z|`` is small) — this matches v0.1.0's behaviour
        # exactly and is exact at any moderate ``z``, including ``|z|`` close
        # to 1 where the truncated Taylor series diverges from the closed
        # form.
        #
        # NOTE: this expression contains an explicit ``/z``.  At very small
        # ``|z|`` the value is fine (``log1p(-z) ≈ -z``, so the ratio is ≈ 1),
        # but JAX's auto-diff of ``-log1p(-z)/z`` cascades ``1/z``, ``1/z²``
        # at higher orders and overflows when ``|z|`` is tiny — same regime
        # as v0.1.0.  In jaxvacua's F_inst chain (parent ``s=3``), only
        # ``_Li_over_z(z, s=2, ...)`` is invoked, so this branch is never
        # hit during higher-order autodiff and the overflow is avoided.
        #
        # v0.3.0: route through :func:`_Li1_over_z_stable`, which uses a
        # 60-term Taylor series at |z|<0.5 so even *direct* high-order
        # autodiff of Li_2 (or anything that ultimately calls Li_1/z at
        # small |z|) stays at fp64 precision.  Equivalent to the closed
        # form at |z|≥0.5, so existing callers see no numerical change.
        return _Li1_over_z_stable(z)
    elif s == 0:
        return 1.0 / (1.0 - z)
    elif s == -1:
        return 1.0 / (1.0 - z)**2
    elif s == -2:
        return (1.0 + z) / (1.0 - z)**3
    elif s == -3:
        return (1.0 + 4.0*z + z**2) / (-1.0 + z)**4
    elif s == -4:
        return (1.0 + 11.0*z + 11.0*z**2 + z**3) / (1.0 - z)**5
    elif s == -5:
        return (1.0 + 26.0*z + 66.0*z**2 + 26.0*z**3 + z**4) / (-1.0 + z)**6
    elif s == -6:
        return (1.0 + 57.0*z + 302.0*z**2 + 302.0*z**3 + 57.0*z**4 + z**5) / (1.0 - z)**7
    elif s == -7:
        return (1.0 + 120.0*z + 1191.0*z**2 + 2416.0*z**3 + 1191.0*z**4 + 120.0*z**5 + z**6) / (-1.0 + z)**8
    elif s == -8:
        return (1.0 + 247.0*z + 4293.0*z**2 + 15619.0*z**3 + 15619.0*z**4 + 4293.0*z**5 + 247.0*z**6 + z**7) / (1.0 - z)**9
    elif s == -9:
        return (1.0 + 502.0*z + 14608.0*z**2 + 88234.0*z**3 + 156190.0*z**4 + 88234.0*z**5 + 14608.0*z**6 + 502.0*z**7 + z**8) / (-1.0 + z)**10
    else:
        if approx == "inf":
            # Re-indexed inf-series: Li_s(z)/z = sum_{k=1}^{p_range-1} z^{k-1}/k^s
            # Hand-extract the k=1 constant term and unroll the rest with
            # static integer exponents (see the ``s==1`` branch above for
            # rationale).
            result = 1.0 + 0.0 * z   # k=1 term, broadcast to z's shape/dtype
            for k in range(2, p_range):
                result = result + z**(k - 1) / float(k)**s
            return result
        elif approx == "integral":
            # Li_s(z)/z = (-1)^{s-1}/Γ(s) · ∫₀¹ log(t)^{s-1}/(1-z·t) dt
            polylog_range = jnp.linspace(0+1e-20, 1., p_range)
            return ((-1)**(s-1) / jax.scipy.special.gamma(s)
                    * jax.scipy.integrate.trapezoid(intgrand(z, polylog_range, s),
                                                    x=polylog_range))
        elif approx == "zero":
            # The "zero" expansion is convergent for |log z| < 2π, i.e. z near 1.
            # In that regime z is not tiny, so dividing by z is safe and the
            # 1/z^n cascade does not occur.  Use the literal form.
            return jax_polylog(z, s, p_range, "zero", pval) / z
        elif approx == "patch":
            # Double-where dispatch — mirror of :func:`jax_polylog`'s patch
            # branch.  Each side sees a safe argument when inactive, so
            # ``0 · NaN = NaN`` poisoning cannot reach the ``jnp.where``.
            # The zero branch divides by z, which is safe because the safe
            # substitute SAFE_ZERO = 0.9 keeps the divisor away from 0.
            mu = jnp.log(z)
            t  = jnp.abs(mu) / (2.0 * jnp.pi)
            use_inf = (jnp.abs(z) < 1.0) & (t >= pval)

            SAFE_INF  = jnp.asarray(0.5 + 0.0j, dtype=z.dtype)
            SAFE_ZERO = jnp.asarray(0.9 + 0.0j, dtype=z.dtype)
            z_for_inf  = jnp.where(use_inf, z, SAFE_INF)
            z_for_zero = jnp.where(use_inf, SAFE_ZERO, z)

            Lis_inf_over_z = 1.0 + 0.0 * z_for_inf
            for k in range(2, p_range):
                Lis_inf_over_z = Lis_inf_over_z + z_for_inf ** (k - 1) / float(k) ** s
            Lis_zero_over_z = jax_polylog(z_for_zero, s, p_range, "zero", pval) / z_for_zero
            return jnp.where(use_inf, Lis_inf_over_z, Lis_zero_over_z)

@jax_polylog.defjvp
def _jax_polylog_jvp(s: int, p_range: int, approx: str, pval: float, primals: tuple, tangents: tuple) -> tuple:
    r"""
    **Description:**
    Forward-mode JVP rule for :func:`jax_polylog`.

    Implements the analytic identity

    .. math::
        \frac{\mathrm{d}}{\mathrm{d}z}\,\mathrm{Li}_s(z) = \frac{\mathrm{Li}_{s-1}(z)}{z}\,,

    with :math:`\mathrm{Li}_{s-1}(z)/z` evaluated via :func:`_Li_over_z`,
    a numerically-stable rewriting that avoids dividing by ``z``.  Using
    ``custom_jvp`` (rather than ``custom_vjp``) also makes
    :func:`jax.jvp` work directly, and JAX automatically derives the VJP
    via transposition (the rule is linear in the tangent).

    Args:
        s, p_range, approx, pval: nondiff_argnums passed through.
        primals (tuple): ``(z,)``.
        tangents (tuple): ``(z_dot,)``.

    Returns:
        tuple: ``(primal_out, tangent_out)`` with
            ``tangent_out = (Li_{s-1}(z)/z) · z_dot``.
    """
    z, = primals
    z_dot, = tangents
    primal_out  = jax_polylog(z, s, p_range, approx, pval)
    deriv       = _Li_over_z(z, s - 1, p_range, approx, pval)
    tangent_out = deriv * z_dot
    return primal_out, tangent_out

jax_polylog_vmap_tmp = jax.vmap(jax_polylog, in_axes=(0, None, None, None, None))

@partial(jit, static_argnames=['s','p_range','approx','pval'])
def jax_polylog_vmap(z: complex, s: int, p_range: int, approx: str = "inf", pval: float = _PVAL_OPTIMAL) -> complex:
    r"""
    **Description:**
    Vectorized version of the polylogarithm function using JAX's vmap.

    Args:
        z (complex): The input values at which to evaluate the polylogarithm. Must be a 1D array.
        s (int): The order of the polylogarithm. Must be an integer.
        p_range (int): The number of terms to include in the series expansion for non-predefined `s` values. Higher values increase accuracy but also computation time.
        approx (str, optional): The approximation method to use. Must be one of "inf", "integral", "zero", or "patch".  Default is ``"inf"``.
        pval (float, optional): Transition parameter for the ``"patch"`` method.  See :func:`jax_polylog`.  Default is :data:`_PVAL_OPTIMAL`.

    Returns:
        complex: The computed polylogarithm values at the input `z`.
    """

    return jax_polylog_vmap_tmp(z, s, p_range, approx, pval)

























