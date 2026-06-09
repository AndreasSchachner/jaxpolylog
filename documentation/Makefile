# Minimal Makefile for Sphinx documentation, mirroring the canonical
# upstream pattern.
#
# Usage:
#   make help       -> list every target Sphinx supports
#   make html       -> build the HTML documentation into build/html
#   make clean      -> remove the entire build directory
#
# `html` (and every other Sphinx target) is dispatched through the
# catch-all rule at the bottom.  Do NOT list those targets under
# .PHONY -- adding them to .PHONY without an explicit recipe makes
# Make treat them as "satisfied" and emit
#     make: Nothing to be done for `html'.

SPHINXOPTS    ?=
# Use ``python -m sphinx`` rather than the bare ``sphinx-build`` binary
# so the build always tracks the active virtual environment.  Override
# with e.g. ``PYTHON=/path/to/venv/bin/python make html``.
PYTHON        ?= python
SPHINXBUILD   ?= $(PYTHON) -m sphinx
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help Makefile clean

# Default target: list what Sphinx can build.
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

# Catch-all: route any other target (html, latexpdf, linkcheck, ...)
# through ``sphinx-build -M``.  The dependency on the Makefile itself
# guarantees this rule is always considered, never short-circuited.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
