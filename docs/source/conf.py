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

from datetime import datetime, timezone

from lightworks import __version__ as ver

project = "Lightworks"
project_copyright = f"{datetime.now(tz=timezone.utc).year}, Aegiq Ltd."  # noqa: UP017
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
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Aegiq/lightworks",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # noqa: E501
            "class": "",
        },
    ],
}
html_static_path = []
html_logo = "assets/lightworks_logo.svg"
html_show_sphinx = True
html_scaled_image_link = False

# Napoleon settings
napoleon_google_docstring = True

# Bibtex
bibtex_bibfiles = ["refs.bib"]
