# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from pathlib import Path

# Add the project root to the Python path so autodoc can import ``purc``.
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# PyTorch and the vendored LaplacianSolve each ship an OpenMP runtime; let them
# share one (avoids the macOS duplicate-libomp abort when autodoc imports purc).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# -- Project information -----------------------------------------------------

project = "PURCSolver"
copyright = "2026, Rui Yao"
author = "Rui Yao"

try:
    from purc import __version__ as release
except Exception:  # pragma: no cover - allow building without the package importable
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]

# Napoleon settings for Google-style docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}

# The package is imported for autodoc (it is installed editable, with the native
# cores built).  Only the optional CVXPY oracle is mocked so the docs build even
# when it is absent.  On a docs-only CI (e.g. ReadTheDocs) without the compiled
# extensions, also add "purc.static_purc._static_purc_core" and
# "purc.laplaciansolve._laplaciansolve_core" here.
autodoc_mock_imports = ["cvxpy"]

autosummary_generate = True

# Intersphinx mapping.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# MathJax.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_theme_options = {
    "navigation_with_keys": True,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "PURCSolver"

# -- Extension configuration -------------------------------------------------

nitpicky = False
nitpick_ignore = [
    ("py:class", "torch.Tensor"),
    ("py:class", "torch.device"),
    ("py:class", "torch.dtype"),
]
