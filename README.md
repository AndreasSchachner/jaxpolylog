# jaxpolylog -- Differentiable polylogarithms in JAX

> *Polylogarithms that JIT-compile, vectorise, and survive arbitrary-order autodiff -- including at $|z| = 10^{-30}$ where the textbook expression has lost every bit of precision.*

<p align="center">
    <a href="https://jaxpolylog.readthedocs.io"><img src="https://readthedocs.org/projects/jaxpolylog/badge/?version=latest" alt="Docs"/></a>
    <a href="https://pypi.org/project/jaxpolylog/"><img src="https://img.shields.io/pypi/v/jaxpolylog.svg" alt="PyPI"/></a>
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python"/></a>
    <a href="https://github.com/AndreasSchachner/jaxpolylog/actions/workflows/ci.yml"><img src="https://github.com/AndreasSchachner/jaxpolylog/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
    <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"/></a>
</p>


**jaxpolylog** is a JAX-native implementation of the polylogarithm

$$\mathrm{Li}_s(z) \;=\; \sum_{k=1}^{\infty}\frac{z^k}{k^s}$$

for **integer order** $s$ and **complex argument** $z$, designed so that the function evaluates, JIT-compiles, vectorises, and **differentiates to arbitrary order under `jax.grad` / `jax.jvp`** without losing fp64 precision -- including in the most numerically aggressive regimes where the textbook series simultaneously diverges and the closed-form expression cancels by 60 orders of magnitude.


## Why this package matters

Polylogarithms appear in nearly every advanced calculation in theoretical physics and number theory: Feynman-integral coefficients, modular forms, period integrals on Calabi-Yau manifolds, instanton sums in string theory, finite-temperature equations of state, and the multi-polylog basis of multi-loop QFT. None of the standard JAX or NumPy/SciPy stacks ship a usable polylogarithm:

- `scipy.special` has **no polylogarithm**; `mpmath.polylog` does, but it is pure Python, not vectorisable, and not differentiable.
- `jax.scipy.special.zeta` covers only $z=1$.
- A naive `jnp.sum` of $\sum z^k/k^s$ is unusable: it diverges for $|z|\ge 1$, loses all precision near $z=1$, and -- most importantly -- its *higher derivatives* under `jax.grad` cascade $1/z^n$ factors that overflow fp64 the moment $|z|$ drops below $\sim 10^{-2}$.

`jaxpolylog` solves all three problems simultaneously and is, to the author's knowledge, the only library that does so:

1. **Two complementary series, glued at the optimal crossover.**
   The convergent series at $|z|<1$ and the Laurent expansion in $\mu = \log z$ about $z = 1$ are combined by an `approx="patch"` dispatch that selects the locally-faster branch. The crossover $t_\star \approx 0.2322$ is the unique fixed point of $e^{-2\pi t}=t$, where the truncation errors of both series are equal -- pre-computed once per import.

2. **Exact Bernoulli arithmetic for the $z\to 1$ expansion.**
   The "zero" expansion coefficients $\zeta(s-k)/k!$ are built from exact `fractions.Fraction` Bernoulli numbers via the Akiyama-Tanigawa recurrence, then floated. The naive `bernoulli(p_range)` route overflows fp64 for $p_{\mathrm{range}} \gtrsim 250$ because individual $B_n$ grow factorially even though $B_n/n!$ decays geometrically; the rational route hits no overflow up to the hard cap of 200 terms, where the truncation error is already $\le 10^{-60}$.

3. **A custom forward-mode JVP using the analytic identity** $\tfrac{\mathrm{d}}{\mathrm{d}z}\mathrm{Li}_s(z) = \mathrm{Li}_{s-1}(z)/z$, evaluated through a stable `_Li_over_z` helper that *never divides by $z$* in the series regime and uses a 60-term Taylor expansion for $\mathrm{Li}_1(z)/z = -\log(1-z)/z$ at small $|z|$. This is what lets the library survive third- and fourth-order autodiff at $|z| \le 10^{-30}$ -- the regime that occurs in string-compactification period integrals where $|z|\sim e^{-2\pi q\cdot\mathrm{Im}(\tau)}$ may genuinely be that small.

4. **Branch safety under `vmap`.**
   The "patch" dispatch uses the *double-where* idiom: both branches are evaluated on safe substitute arguments when inactive, so that batched JIT compilation with mixed-regime inputs cannot leak `0 * NaN = NaN` into the result.

5. **Closed forms for $s \le 1$** are hard-coded, so neutral orders ($s = 0, -1, -2, \dots, -9$) reduce to rational functions in $z$ and round-trip exactly through autodiff.

The result: $\mathrm{Li}_s$ -- and all of its derivatives, up to whatever order JAX is willing to trace -- become a single JIT-traceable, vmap-traceable, autodiff-clean primitive that downstream packages (period computations, flux-vacuum scans, conifold expansions, instanton sums) can drop into their forward and backward passes without writing any special-case code. `jaxpolylog` is what makes [`jaxvacua`](https://github.com/AndreasSchachner/jaxvacua) and the broader [`stringforge`](https://github.com/AndreasSchachner/stringforge) ecosystem able to autodiff through the Kahler potential and the GVW superpotential all the way to fourth order in the deep LCS limit.


## Installation

You may want to install the code in a new virtual environment, which can be created via `python -m venv jaxpolylog-env` and activated with `source jaxpolylog-env/bin/activate` from within the terminal at a desired working directory.

> [!NOTE]
> If a specific build of [JAX](https://github.com/jax-ml/jax) is required (e.g. with GPU support), follow the JAX installation instructions [here](https://github.com/jax-ml/jax#installation) before installing `jaxpolylog`. Otherwise, the default CPU build of JAX is pulled in automatically.

The recommended way to install the package is via [pip](https://packaging.python.org/en/latest/key_projects/#pip). Before installing, make sure your packaging tools are up to date by running

`pip install --upgrade pip setuptools`

Next, choose the installation method that best fits your use case:

- **Install from PyPI (recommended):**
  `pip install jaxpolylog`

- **Editable install directly from GitHub:**
  `pip install -e git+https://github.com/AndreasSchachner/jaxpolylog.git#egg=jaxpolylog`

- **Editable install from a local clone (recommended for development):**
  After cloning the repository, run `pip install -e .` from the project root.
  The `-e` (editable) flag ensures that local code changes take effect immediately without requiring reinstallation.


## Requirements

The code currently supports Python `>= 3.12`. The required packages -- `numpy`, `jax`, `jaxlib` -- are listed in `setup.py` and are pulled in automatically during installation.


## Documentation

The documentation is generated with [Sphinx](https://www.sphinx-doc.org/en/master/). After installing `jaxpolylog`, install the additional documentation requirements from [`documentation/requirements.txt`](documentation/requirements.txt) via

`pip install -r documentation/requirements.txt`

and then build the HTML output with `cd documentation && make html`. The rendered HTML lives in [`documentation/build/html`](documentation/build/html).


## Quick start

```python
import jax
import jax.numpy as jnp
from jaxpolylog import jax_polylog, jax_polylog_vmap

# Evaluate Li_3 at a single point using the auto-patched series
z = jnp.array(0.7 + 0.1j)
val = jax_polylog(z, s=3, p_range=200, approx="patch")

# Vectorised evaluation along an axis
zs = jnp.linspace(0.1, 0.95, 32) + 0.0j
vals = jax_polylog_vmap(zs, s=2, p_range=200, approx="patch")

# Forward- and reverse-mode autodiff, to arbitrary order
dLi3_dz   = jax.grad(lambda z: jax_polylog(z, 3, 200, "patch").real)(0.5 + 0.0j)
d2Li3_dz2 = jax.grad(jax.grad(
                lambda z: jax_polylog(z, 3, 200, "patch").real))(0.5 + 0.0j)
```

Pick the `approx` strategy that matches the regime:

| `approx`     | Regime where it is optimal               | Notes                                   |
|--------------|------------------------------------------|-----------------------------------------|
| `"inf"`      | Mid-range $\|z\|$ inside the unit disk   | Convergent series $\sum z^k/k^s$        |
| `"zero"`     | $z$ near $1$ (i.e. $\|\log z\| < 2\pi$)  | Laurent series in $\mu = \log z$        |
| `"patch"`    | **Default for general use**              | Auto-dispatches at the fixed point      |
| `"integral"` | Reference/verification                   | Trapezoid quadrature, slower            |


## Tests

The repository ships with **896 tests** across three suites covering closed-form values, primal accuracy against `mpmath` to twelve digits, first- through third-order derivative identities, vectorised `vmap` accuracy, the `patch` regression cases, and high-order autodiff stability in the deep-LCS regime. Run them with

```bash
pip install pytest mpmath
pytest tests/ -q
```


## Repository structure

```
.
|-- jaxpolylog/                    # main package
|   |-- __init__.py                # re-exports `jax_polylog`, `jax_polylog_vmap`
|   `-- polylogs.py                # series, JVP rule, vmap wrapper, stability helpers
|-- tests/                         # 896 tests across three suites
|   |-- test_polylog_accuracy.py
|   |-- test_polylog_derivatives.py
|   `-- test_patch_high_deriv_vmap.py
|-- documentation/                 # Sphinx documentation source
|-- LICENSE
|-- README.md
`-- setup.py
```


## Citation

If you use `jaxpolylog` in academic work, please cite the parent project for which it was developed,

```bibtex
@article{Dubey:2023dvu,
    author        = "Dubey, Abhishek and Krippendorf, Sven and Schachner, Andreas",
    title         = "{JAXVacua --- a framework for sampling string vacua}",
    eprint        = "2306.06160",
    archivePrefix = "arXiv",
    primaryClass  = "hep-th",
}
```


## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).


## Contact

For questions or feedback, please get in touch: <as3475@cornell.edu>, or open an issue at <https://github.com/AndreasSchachner/jaxpolylog/issues>.
