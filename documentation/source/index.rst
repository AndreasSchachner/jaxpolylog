jaxpolylog -- Polylogarithms in JAX with autodiff
==================================================

**jaxpolylog** is a JAX-native implementation of the polylogarithm

.. math::

    \mathrm{Li}_s(z) \;=\; \sum_{k=1}^{\infty}\frac{z^k}{k^s}

for integer order :math:`s` and complex argument :math:`z`, designed
so that the function evaluates, JIT-compiles, vectorises, and
**differentiates to arbitrary order** under :func:`jax.grad` /
:func:`jax.jvp` without losing fp64 precision -- including in the
deep-LCS regimes where :math:`|z|` may be as small as
:math:`10^{-30}` or below.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: New here?
      :link: intro
      :link-type: doc

      Start with the introduction.  It explains the polylogarithm
      and the four numerical strategies (``"inf"``, ``"zero"``,
      ``"patch"``, ``"integral"``) the package implements.

   .. grid-item-card:: Looking for the API?
      :link: jaxpolylog
      :link-type: doc

      The API reference documents the public functions
      :func:`jax_polylog` and :func:`jax_polylog_vmap` and the
      custom JVP rule underneath.


Why is this package important?
------------------------------

Polylogarithms appear in nearly every advanced calculation in
theoretical physics and number theory: Feynman-integral coefficients,
modular forms, period integrals on Calabi--Yau manifolds, instanton
sums in string theory, finite-temperature equations of state, and the
multi-polylog basis of multi-loop QFT.  None of the standard JAX or
NumPy/SciPy stacks ship a usable polylogarithm:

* :mod:`scipy.special` has **no polylogarithm**;
  :func:`mpmath.polylog` does, but it is pure Python, not
  vectorisable, and not differentiable.
* :func:`jax.scipy.special.zeta` covers only :math:`z = 1`.
* A naive ``jnp.sum`` of :math:`\sum z^k / k^s` is unusable: it
  diverges for :math:`|z| \ge 1`, loses all precision near
  :math:`z = 1`, and -- most importantly -- its *higher derivatives*
  under :func:`jax.grad` cascade :math:`1/z^n` factors that
  overflow fp64 the moment :math:`|z|` drops below
  :math:`\sim 10^{-2}`.

``jaxpolylog`` solves all three problems at once.  See
:doc:`intro` for the full numerical story.


Quick start
-----------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jaxpolylog import jax_polylog, jax_polylog_vmap

   # Evaluate Li_3 at a single point using the auto-patched series
   z   = jnp.array(0.7 + 0.1j)
   val = jax_polylog(z, s=3, p_range=200, approx="patch")

   # Vectorised evaluation along an axis
   zs   = jnp.linspace(0.1, 0.95, 32) + 0.0j
   vals = jax_polylog_vmap(zs, s=2, p_range=200, approx="patch")

   # Forward- and reverse-mode autodiff, to arbitrary order
   dLi3   = jax.grad(lambda z: jax_polylog(z, 3, 200, "patch").real)(0.5 + 0.0j)
   d2Li3  = jax.grad(jax.grad(
                lambda z: jax_polylog(z, 3, 200, "patch").real))(0.5 + 0.0j)


.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   intro
   notebooks/quickstart
   jaxpolylog


Indices and search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
