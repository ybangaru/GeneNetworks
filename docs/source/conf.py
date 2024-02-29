# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


project = 'GraphXL'
copyright = '2024, GraphXL'
author = 'ybangaru <yaswanthbangaru13@gmail.com>'
release = '0.0.1alpha'


extensions = ["sphinx_rtd_theme", "sphinx.ext.todo", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

