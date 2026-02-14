# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

autodoc_mock_imports = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.utils',
    'torch.utils.checkpoint',
    'numpy',
    'einops',
    'causal_conv1d',
    'mamba_ssm',
    'triton',
    'opt_einsum',
    "selective_scan_cuda",
    "lrnnx.ops.triton"
]


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lrnnx'
copyright = '2026, Karan Bania, Soham Kalburgi, Manit Tanwar, Dhruthi, Aditya Nagarsekar, Harshvardhan Mestha, Naman Chibber, Raj Deshmukh, Anish Sathyanarayanan, Aarush Rathore, Pratham Chheda'
author = 'Karan Bania, Soham Kalburgi, Manit Tanwar, Dhruthi, Aditya Nagarsekar, Harshvardhan Mestha, Naman Chibber, Raj Deshmukh, Anish Sathyanarayanan, Aarush Rathore, Pratham Chheda'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}
