# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import os
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime

from lightworks import __version__ as ver

project = "Lightworks"
project_copyright = f"{datetime.now(tz=datetime.timezone.utc).year}, Aegiq Ltd."
author = "Aegiq Ltd."
release = ".".join(ver.split(".")[:2])
version = ver

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = []

# Set python path variable to enable nbsphinx to run notebooks if required
os.environ["PYTHONPATH"] = os.path.abspath("../../../lightworks")  # noqa: PTH100

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "Documentation"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#3e368d",
        "color-brand-content": "#873bd4",
    },
    "dark_css_variables": {
        "color-brand-primary": "#9198cc",
        "color-brand-content": "#9198cc",
    },
}
html_static_path = []
html_logo = "assets/lightworks_logo.svg"
html_show_sphinx = True
html_scaled_image_link = False

# Napoleon settings
napoleon_google_docstring = True

# Bibtex
bibtex_bibfiles = ["refs.bib"]
