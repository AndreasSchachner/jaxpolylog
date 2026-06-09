"""Sphinx configuration for the jaxpolylog documentation.

Mirrors the layout used by JAXVacua: ``sphinx_book_theme`` HTML
output, ``autodoc`` + ``autosummary`` for the API reference,
``napoleon`` for Google-style docstrings, ``myst_nb`` for Markdown
and notebook pages, and ``sphinx_copybutton`` / ``sphinx_togglebutton``
/ ``sphinx_design`` for the interactive components.
"""

# Configuration file for the Sphinx documentation builder.
# For a full list of options see
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys
from pathlib import Path

# Make the local jaxpolylog package importable for autodoc.
sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------
def _read_package_version() -> str:
    """Read ``__version__`` from ``jaxpolylog/__init__.py`` without importing."""
    init_py = Path(__file__).resolve().parents[2] / "jaxpolylog" / "__init__.py"
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", init_py.read_text(), re.M)
    return match.group(1) if match else "0.0.0"


project = "jaxpolylog"
copyright = "2022-2026, Andreas Schachner"
author = "Andreas Schachner"

release = _read_package_version()
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------
# NOTE: ``sphinx_autodoc_typehints`` is *not* enabled here.  ``jax_polylog``
# is wrapped by ``jax.custom_jvp`` + ``jax.jit``, and the extension cannot
# align type hints with the wrapped signature -- it raises
# ``list assignment index out of range``.  Stock ``sphinx.ext.autodoc``
# renders the (annotated) signature without issue.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_design",
    "myst_nb",
]

templates_path = ["_templates"]
source_suffix = {
    ".rst":   "restructuredtext",
    ".md":    "myst-nb",
    ".ipynb": "myst-nb",
}
exclude_patterns = []

# autosummary generates stub pages with their own ``.. autofunction::``,
# so we lean on it alone -- ``jaxpolylog.rst`` does NOT call
# ``.. autofunction::`` on the same names (which would emit
# ``duplicate object description`` warnings).
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_use_rtype = False
napoleon_custom_sections = [("Returns", "params_style")]
add_module_names = False
toc_object_entries_show_parents = "hide"

# Tolerate the cosmetic warnings that come from documenting a
# ``custom_jvp``-decorated function and from docstrings that contain
# ``|z|`` (docutils tries to read it as a substitution reference).
# Neither blocks the rendered HTML.
suppress_warnings = [
    "autodoc",
    "docutils",
]


# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/AndreasSchachner/jaxpolylog",
    "repository_branch": "main",
    "path_to_docs": "documentation/source",
    "use_repository_button": True,
    "use_edit_page_button": True,
}
html_static_path = ["_static"]


# -- MyST / myst_nb ----------------------------------------------------------
myst_enable_extensions = ["dollarmath", "deflist", "colon_fence"]
myst_heading_anchors = 4
nb_execution_mode = "off"            # do not execute notebooks on the RTD build
nb_execution_allow_errors = False
nb_merge_streams = True
nb_execution_timeout = 120


# -- MathJax 3 ---------------------------------------------------------------
mathjax3_config = {
    "tex": {"inlineMath": [["$", "$"], ["\\(", "\\)"]]},
    "svg": {"fontCache": "global"},
}
