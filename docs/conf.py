import os
import sys

import sphinx_book_theme

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("../src/mjlab"))


project = "mjlab"
copyright = "2025, The mjlab Developers"
author = "The mjlab Developers"

extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.autosummary",
  "autodocsumm",
  "myst_parser",
  "sphinx.ext.napoleon",
  "sphinxemoji.sphinxemoji",
  "sphinx.ext.intersphinx",
  "sphinx.ext.mathjax",
  "sphinx.ext.todo",
  "sphinx.ext.viewcode",
  "sphinxcontrib.bibtex",
  "sphinxcontrib.icon",
  "sphinx_copybutton",
  "sphinx_design",
  "sphinx_tabs.tabs",
  "sphinx_multiversion",
]

mathjax3_config = {
  "tex": {
    "inlineMath": [["\\(", "\\)"]],
    "displayMath": [["\\[", "\\]"]],
  },
}

panels_add_bootstrap_css = False
panels_add_fontawesome_css = True

source_suffix = {
  ".rst": "restructuredtext",
  ".md": "markdown",
}

nitpick_ignore = [
  ("py:obj", "slice(None)"),
]

nitpick_ignore_regex = [
  (r"py:.*", r"pxr.*"),
  (r"py:.*", r"trimesh.*"),
]

# emoji style
sphinxemoji_style = "twemoji"
autodoc_typehints = "signature"
autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True
bibtex_bibfiles = ["source/_static/refs.bib"]
autosummary_generate = True
autosummary_generate_overwrite = False
autodoc_default_options = {
  "members": True,
  "undoc-members": True,
  "show-inheritance": True,
  "member-order": "bysource",
  "autosummary": True,
}
intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
}

exclude_patterns = [
  "_build",
  "_redirect",
  "_templates",
  "Thumbs.db",
  ".DS_Store",
  "README.md",
  "licenses/*",
]

autodoc_mock_imports = [
  "matplotlib",
  "scipy",
  "carb",
  "warp",
  "pxr",
  "h5py",
  "hid",
  "prettytable",
  "tqdm",
  "tensordict",
  "trimesh",
  "toml",
  "mujoco_warp",
  "gymnasium",
  "rsl_rl",
  "viser",
  "wandb",
  "torchvision",
]

suppress_warnings = [
  "ref.python",
  "docutils",
]

language = "en"

html_title = "mjlab Documentation"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"
html_favicon = "source/_static/favicon.ico"
html_show_copyright = True
html_show_sphinx = False
html_last_updated_fmt = ""

html_static_path = ["source/_static"]
html_css_files = ["custom.css"]

html_theme_options = {
  "path_to_docs": "docs/",
  "collapse_navigation": True,
  "repository_url": "https://github.com/mujocolab/mjlab",
  "use_repository_button": True,
  "use_issues_button": True,
  "use_edit_page_button": True,
  "show_toc_level": 2,
  "use_sidenotes": True,
  "logo": {
    "text": "The mjlab Documentation",
  },
  "icon_links": [
    {
      "name": "GitHub",
      "url": "https://github.com/mujocolab/mjlab",
      "icon": "fa-brands fa-square-github",
      "type": "fontawesome",
    },
    {
      "name": "mjlab",
      "url": "https://github.com/mujocolab/mjlab",
      "icon": "https://img.shields.io/badge/mjlab-1.1.0-silver.svg",
      "type": "url",
    },
    {
      "name": "Benchmarks",
      "url": "https://mujocolab.github.io/mjlab/nightly/",
      "icon": "https://img.shields.io/badge/Nightly-Benchmarks-blue",
      "type": "url",
    },
    {
      "name": "Stars",
      "url": "https://img.shields.io/github/stars/mujocolab/mjlab?color=fedcba",
      "icon": "https://img.shields.io/github/stars/mujocolab/mjlab?color=fedcba",
      "type": "url",
    },
  ],
  "icon_links_label": "Quick Links",
}

templates_path = [
  "_templates",
]

smv_remote_whitelist = r"^.*$"
smv_branch_whitelist = os.getenv("SMV_BRANCH_WHITELIST", r"^(main|devel)$")
smv_tag_whitelist = os.getenv("SMV_TAG_WHITELIST", r"^v[1-9]\d*\.\d+\.\d+$")

html_sidebars = {
  "**": [
    "navbar-logo.html",
    "icon-links.html",
    "search-field.html",
    "sbt-sidebar-nav.html",
  ]
}


def skip_member(app, what, name, obj, skip, options):
  exclusions = ["from_dict", "to_dict", "replace", "copy", "validate", "__post_init__"]
  if name in exclusions:
    return True
  return None


def setup(app):
  app.connect("autodoc-skip-member", skip_member)
