"""Sphinx configuration."""
project = "Cookiecutter Hypermodern Python ML Example"
author = "Matthew Sach"
copyright = "2022, Matthew Sach"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser"]
autodoc_typehints = "description"
html_theme = "furo"
