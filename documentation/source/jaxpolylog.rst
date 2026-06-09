.. currentmodule:: jaxpolylog


API reference
=============

The full public surface of ``jaxpolylog`` lives in a single module,
``jaxpolylog.polylogs``, and is re-exported through the top-level
``jaxpolylog`` package.


Top-level functions
-------------------

.. autosummary::
   :toctree: _autosummary

   jax_polylog
   jax_polylog_vmap


Numerical and algorithmic helpers
---------------------------------

Most users will only call :func:`jax_polylog` and
:func:`jax_polylog_vmap`.  The helpers below document the
implementation choices behind the scenes -- the exact Bernoulli
construction of the ``"zero"`` expansion coefficients, the
custom-JVP rule, and the small-:math:`|z|`-safe form of
:math:`\mathrm{Li}_1(z)/z`.  They are surfaced here so that downstream
authors can reason about precision and autodiff regimes without
having to read the source.

.. autosummary::
   :toctree: _autosummary

   polylogs._zero_branch_coeffs
   polylogs._zeta_pos_int
   polylogs._Li1_over_z_stable
   polylogs._compute_pval_optimal
   polylogs._Li_over_z


Optimal patch crossover
-----------------------

.. autodata:: jaxpolylog.polylogs._PVAL_OPTIMAL
   :no-value:

The unique positive solution of :math:`e^{-2\pi t} = t`, used as the
default crossover between the ``"inf"`` and ``"zero"`` series in
``approx="patch"``.  Approximately :math:`0.2322`.  Precomputed once
at import time by bisection.
