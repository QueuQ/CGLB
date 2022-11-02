# Configuration file for the Sphinx documentation builder.

import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

autodoc_mock_imports = ["torch","dgl","copy","numpy","sklearn","ogb","quadprog","dgllife","matplotlib"]

# -- Project information

project = 'CGLB'
copyright = '2022, ZHANG Xikun'
author = 'ZHANG Xikun'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'