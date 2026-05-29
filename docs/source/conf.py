# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'profsea-climate'
copyright = '2026, MetOffice'
author = 'isabellaascione'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_mock_imports = [
    "cartopy",
    "dask",
    "fair",
    "matplotlib",
    "numcodecs",
    "numpy",
    "pandas",
    "profsea.emulator",
    "profsea.plotting_libraries",
    "requests",
    "rich",
    "rich_argparse",
    "scipy",
    "xarray",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'show_nav_level': 2,
    'secondary_sidebar_items': [],
}
html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs", "page-toc"],
}
html_static_path = ['_static']
