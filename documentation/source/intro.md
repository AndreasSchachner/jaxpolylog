# Introduction

`jaxpolylog` provides a single primitive:

$$
\mathrm{Li}_s(z) \;=\; \sum_{k=1}^{\infty}\frac{z^{k}}{k^{s}} ,
$$

for integer order $s$ and complex argument $z$, evaluated through a
JAX-traceable code path that supports JIT compilation, `vmap`, and
arbitrary-order forward- and reverse-mode automatic differentiation.

## Mathematical background

For $|z| < 1$ the series defines $\mathrm{Li}_s(z)$ directly. The
function admits the integral representation

$$
\mathrm{Li}_s(z) \;=\; \frac{(-1)^{s-1}}{\Gamma(s)}
\int_{0}^{1} \frac{\bigl(\log t\bigr)^{s-1}}{1 - z\,t}\, \mathrm{d}t ,
$$

and analytically continues to the cut plane $\mathbb{C}\setminus [1,\infty)$.
Closed-form expressions in $z$ exist for $s \le 1$ -- for instance,

$$
\mathrm{Li}_1(z) = -\log(1-z), \qquad
\mathrm{Li}_0(z) = \frac{z}{1-z}, \qquad
\mathrm{Li}_{-1}(z) = \frac{z}{(1-z)^2} ,
$$

and so on through $s = -9$.  `jaxpolylog` hard-codes the closed forms
for $s \in \{1, 0, -1, \dots, -9\}$ so that these neutral orders
round-trip exactly through autodiff.

The package implements four complementary evaluation strategies for
the remaining integer orders $s \ge 2$ (or $s \le -10$ via the general
series), selected by the `approx` keyword.

## The four `approx` strategies

`"inf"`
:   The defining series $\sum_{k=1}^{N-1} z^k / k^s$, truncated at
    `p_range` terms.  Convergent for $|z| < 1$, with truncation
    error $\mathcal{O}\!\left(|z|^{N}\right)$.  Fast and exact at
    moderate $|z|$, but loses every digit of precision as $z \to 1$.

`"zero"`
:   The Laurent expansion in $\mu \equiv \log z$ about $z = 1$,

    $$
    \mathrm{Li}_s(z) \;=\;
    \frac{\mu^{s-1}}{(s-1)!}\!\left(H_{s-1} - \log(-\mu)\right)
    \;+\; \sum_{k=0}^{P-1} \frac{\zeta(s-k)}{k!}\,\mu^{k} ,
    $$

    convergent for $|\mu| < 2\pi$ -- i.e. on the whole disk
    $|z - 1| < 2\pi$ minus the branch cut.  Optimal for $z$ near $1$
    where the `"inf"` series stalls.

`"patch"` *(default for general use)*
:   A double-where dispatch that selects between `"inf"` and
    `"zero"` per input value.  The crossover criterion is

    $$
    |z| < 1 \;\text{ and }\; t \equiv \tfrac{|\log z|}{2\pi} \ge t_{\star}
    \;\;\Longrightarrow\;\; \text{use ``inf''};
    \qquad \text{otherwise use ``zero''}.
    $$

    The fixed point $t_{\star} \approx 0.2322$ is the unique
    positive solution of $e^{-2\pi t} = t$, where the truncation
    errors of the two series are equal.  It is independent of the
    truncation order, so the crossover does not have to be retuned
    when `p_range` changes.

`"integral"`
:   Trapezoid quadrature of the integral representation.  Used as a
    reference for verification and benchmarking; not the path of
    choice for production work.

## Numerical stability under autodiff

The custom JVP rule of {func}`jaxpolylog.jax_polylog` implements the
analytic identity

$$
\frac{\mathrm{d}}{\mathrm{d}z}\,\mathrm{Li}_s(z)
\;=\; \frac{\mathrm{Li}_{s-1}(z)}{z}\, ,
$$

evaluated through a stable `_Li_over_z(z, s-1, ...)` helper that
**never divides by `z` in the series regime**.  For $s \le 0$ each
$\mathrm{Li}_s(z)/z$ collapses to a rational function in $z$ with no
$1/z$ left over; for the `"inf"` and `"zero"` branches the helper
re-indexes the series so that the $1/z$ factor cancels at the level
of the coefficients.  At very small $|z|$ the special case $s = 1$
routes through a 60-term Taylor series of $-\log(1-z)/z$.

The net effect is that third- and fourth-order autodiff of
$\mathrm{Li}_s$ at $|z| \le 10^{-30}$ -- the regime that genuinely
occurs in string-compactification period integrals where
$|z| \sim e^{-2\pi q\cdot\mathrm{Im}\tau}$ may legitimately be that
small -- remains numerically stable at fp64 precision.

## Branch safety under `vmap`

The `"patch"` dispatch uses the *double-where* idiom: both the `"inf"`
and `"zero"` branches are evaluated on safe substitute arguments
(`SAFE_INF = 0.5`, `SAFE_ZERO = 0.9`) when inactive, so the inactive
branch always produces a finite -- but irrelevant -- number that the
final `jnp.where` discards.  This avoids the `0 * NaN = NaN`
poisoning that would otherwise occur whenever a batched JIT
trace evaluates both branches with mixed-regime inputs.

## Exact Bernoulli arithmetic for the `"zero"` expansion

The coefficient $\zeta(s - k) / k!$ in the `"zero"` series is built
from exact `fractions.Fraction` Bernoulli numbers via the
Akiyama-Tanigawa recurrence, then floated.  The naive route
through `jax.scipy.special.bernoulli(p_range)` overflows fp64 for
$p_\text{range} \gtrsim 250$ because individual $B_n$ grow factorially
even though the coefficient $B_n / n!$ that we actually need decays
geometrically.  The rational construction hits no overflow up to the
hard cap of 200 terms, where the truncation error is already
$\le 10^{-60}$.

For positive integer $n \ge 2$ the helper `_zeta_pos_int(n)` uses
the closed form

$$
\zeta(2m) \;=\; \frac{|B_{2m}|\,(2\pi)^{2m}}{2\,(2m)!}
$$

for even $n$ and a 60-term sum plus a five-term Euler-Maclaurin tail
for odd $n$ -- both more than enough for full fp64 precision.

## Tests and guarantees

The repository ships with **896 tests** across three suites:

- `test_polylog_accuracy.py` -- closed-form values, mid-range
  $|z|$, `"zero"` expansion accuracy near $z = 1$, the `"patch"`
  crossover, the `pval` parameter, and `vmap` accuracy.
- `test_polylog_derivatives.py` -- first, second, and third
  derivatives against `mpmath`; the identity
  $\frac{\mathrm{d}}{\mathrm{d}z}\mathrm{Li}_s(z) = \mathrm{Li}_{s-1}(z)/z$
  in closed form; forward-mode autodiff; `vmap` derivatives;
  regression cases.
- `test_patch_high_deriv_vmap.py` -- the v0.3.0 fixes for the
  `"patch"` branch under mixed `vmap` inputs and high-order
  autodiff in the deep-LCS regime.

Run them with `pytest tests/ -q`.
