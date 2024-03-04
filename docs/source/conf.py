
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime
from lightworks import __version__ as ver

project = 'Lightworks'
copyright = f'{datetime.now().year}, Aegiq Ltd.'
author = 'Aegiq Ltd.'
release = ".".join(ver.split(".")[:2])
version = ver

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc', 
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'nbsphinx',
    'sphinxcontrib.bibtex',
]

if False: # Switch this to True to export as pdf
    extensions += ['sphinx.ext.imgconverter']
    image_converter_args=["-density", "300"]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = "Lightworks Documentation"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#3e368d",
        "color-brand-content": "#3e368d",
    },
    "dark_css_variables": {
        "color-brand-primary": "#9198cc",
        "color-brand-content": "#9198cc",
    },
}
html_static_path = ['_static']
html_logo = "assets/aegiq_logo.svg"
html_show_sphinx = True
html_scaled_image_link = False

# Napoleon settings
napoleon_google_docstring = True

# Bibtex
bibtex_bibfiles = ['refs.bib']