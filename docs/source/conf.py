# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # make cvrunner importable

project = 'cvrunner'
copyright = '2025, Long-Giang Vu'
author = 'Long-Giang Vu'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",           # Google/NumPy style support
    "sphinx_autodoc_typehints",     # include type hints
    "sphinx.ext.autosummary",
]

autosummary_generate = True   # automatically create stub pages

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rfutd_theme'
html_static_path = ['_static']
