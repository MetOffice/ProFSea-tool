# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ProFSea"
copyright = "2026, MetOffice"
author = "isabellaascione"
release = "3.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_design",
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

templates_path = ["_templates"]
exclude_patterns = []

nbsphinx_prolog = ""
# Allow nbsphinx to find notebooks outside the docs/source directory
nbsphinx_allow_errors = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "header_links_before_dropdown": 4,
    "secondary_sidebar_items": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MetOffice/ProFSea-tool",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
}
html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs", "page-toc"],
}
html_static_path = ["_static"]
html_logo = "_static/profsea-logo.png"
html_favicon = '_static/favicon.png'
