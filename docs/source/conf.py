# Configuration file for the Sphinx documentation builder.
from pathlib import Path

# resolve package root
ROOT = Path(__file__).resolve().parents[1]
autoapi_dirs = [str(ROOT / "src")]

__doc_const_in_modules__ = [
    "config",
    "field",
]

project = 'quvac'
copyright = '2025, maxbalrog'
author = 'maxbalrog'
release = '0.1.1'

# --------------------- General configuration ---------------------------------

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",]

# autoapi_dirs = ["../../src/quvac"]  # Path to your package
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_ignore = ["*/cluster/*", "*.ipynb_checkpoints/*",]

templates_path = ['_templates']
exclude_patterns = []

def skip_undocumented_members(app, what, name, obj, skip, options):
    # Skip dublicate entries for class attributes and methods present in
    # class docstring and __init__ method
    if what == "attribute":
        skip = True
    # skip all module constants
    if what == "data" and not skip:
        skip = True
        for module in __doc_const_in_modules__:
            if module in name:
                skip = False
    return skip

def setup(app):
    app.connect("autoapi-skip-member", skip_undocumented_members)

# --------------------- Options for HTML output ------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_logo = "_static/logo.jpg"

html_theme_options = {
    "sidebar_hide_name": True,   # hides the project name, keeps only the logo
}
