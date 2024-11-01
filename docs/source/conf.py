# ruff: noqa
from importlib.metadata import version as get_version
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "simpunch"
copyright = "2024, PUNCH Science Operations Center"
author = "PUNCH Science Operations Center"

# The full version, including alpha/beta/rc tags
release: str = get_version("simpunch")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = ["autoapi.extension",
              "sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "sphinx_favicon"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_static_path = ["_static"]
html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/punch-mission/simpunch",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "show_nav_level": 1,
    "show_toc_level": 3,
    "logo": {
        "text": "simpunch",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
}
html_context = {
    "github_user": "punch-mission",
    "github_repo": "simpunch",
    "github_version": "main",
    "doc_path": "docs/source/",
}


autoapi_dirs = ["../../simpunch"]

favicons = ["favicon.ico"]
