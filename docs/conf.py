# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- REQUIREMENTS -----------------------------------------------------
# pip install sphinx-material
# pip install sphinxemoji

import datetime
import os
import re
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------
def find_author():
    """This returns 'The NeuroKit's development team'"""
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format("__author__"),
        open("../neurokit2/__init__.py").read(),
    )
    return str(result.group(1))


project = "NeuroKit2"
copyright = f"2020–{datetime.datetime.now().year}"
author = '<a href="https://dominiquemakowski.github.io/">Dominique Makowski</a> and the <a href="https://github.com/neuropsychology/NeuroKit/blob/master/AUTHORS.rst">Team</a>. This documentation is licensed under a <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> license.'

# The short X.Y version.
def find_version():
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format("__version__"),
        open("../neurokit2/__init__.py").read(),
    )
    return result.group(1)


version = find_version()
# The full version, including alpha/beta/rc tags.
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinxemoji.sphinxemoji",
    "sphinx_copybutton",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# Ignore duplicated sections warning
suppress_warnings = ["epub.duplicated_toc_entry"]
nitpicky = False  # Set to True to get all warnings about crosslinks

# Prefix document path to section labels, to use:
# `path/to/file:heading` instead of just `heading`
autosectionlabel_prefix_document = True

# -- Options for autodoc -------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
add_module_names = False  # If true, the current module name will be prepended to all description

# -- Options for ipython directive  ----------------------------------------

# Doesn't work?
# ipython_promptin = ">"  # "In [%d]:"
# ipython_promptout = ">"  # "Out [%d]:"

# -- Options for myst_nb ---------------------------------------------------
nb_execution_mode = "force"
nb_execution_raise_on_error = True

# googleanalytics_id = "G-DVXSEGN5M9"


# NumPyDoc configuration -----------------------------------------------------

# -- Options for HTML output -------------------------------------------------

html_favicon = "img/icon.ico"
html_logo = "img/neurokit.png"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_book_theme"

# https://sphinx-book-theme.readthedocs.io/en/latest/customize/index.html
html_theme_options = {
    "repository_url": "https://github.com/neuropsychology/NeuroKit",
    "repository_branch": "dev",  # TODO: remove this before merging
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs/",
    "use_edit_page_button": True,
    "logo_only": True,
    "show_toc_level": 1,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
