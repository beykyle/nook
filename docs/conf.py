"""Sphinx configuration for the nook documentation site."""

from importlib.metadata import version as _version

project = "nook"
author = "Kyle Beyer"
copyright = "2026, Kyle Beyer"
release = _version("nook")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build"]
templates_path = ["_templates"]

# The existing pages cross-link with GitHub-style #anchors
# (e.g. limitations.md#the-mass-surface), so generate matching slugs.
myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence"]

autosummary_generate = True
autodoc_member_order = "bysource"
# every module uses `from __future__ import annotations`; keep signatures
# readable by moving type hints into the description
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

html_theme = "furo"
html_title = f"nook {release}"
