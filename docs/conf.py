# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from importlib.metadata import PackageNotFoundError, metadata

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

# Make project code + local Sphinx extensions importable
sys.path[:0] = [str(REPO_ROOT), str(HERE / "extensions")]

# -- Project information -----------------------------------------------------

# Defaults (works even if BRIDGE is NOT installed as a pip package)
project_name = "BRIDGE"
author = "Wang et al."
version = "0.0.0"
release = version

# Optional: if you have an installable distribution, we try to read its metadata.
# If none is installed, we just keep the defaults above.
_info = None
for _dist in ("bridge", "BRIDGE"):
    try:
        _info = metadata(_dist)
        break
    except PackageNotFoundError:
        pass

if _info:
    project_name = _info.get("Name", project_name)
    author = _info.get("Author", author)
    version = _info.get("Version", version)
    release = version

project = project_name
copyright = f"{datetime.now():%Y}, {author}."

# GitHub repo (for theme buttons + source links)
repository_url = "https://github.com/wangyb97/BRIDGE"
github_user = "wangyb97"
github_repo = "BRIDGE"
github_branch_for_edit = "main"  # used for "Edit this page" button

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True
needs_sphinx = "4.0"

html_context = {
    "display_github": True,
    "github_user": github_user,
    "github_repo": github_repo,
    "github_version": github_branch_for_edit,
    "conf_py_path": "/docs/",
}

autodoc_mock_imports = [
    "einops",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "sklearn",
    "skimage",
    "pandas",
    "igrads",
    "torch",
    "torchvision",
    "torch_geometric",
    "triton",
]


# -- Extensions --------------------------------------------------------------
def _maybe(ext: str) -> str | None:
    """Only enable an extension if it is importable."""
    try:
        return ext if importlib.util.find_spec(ext) is not None else None
    except ModuleNotFoundError:
        return None

extensions = [
    e
    for e in [
        "myst_nb",
        "sphinx.ext.autodoc",
        # "sphinx.ext.intersphinx",
        "sphinx.ext.linkcode",  # provides "view source" links via linkcode_resolve below
        "sphinx.ext.mathjax",
        "sphinx.ext.napoleon",
        "sphinx_autodoc_typehints",  # should be after napoleon
        "sphinx.ext.extlinks",
        "sphinx.ext.autosummary",
        "sphinxcontrib.bibtex",
        "sphinx_copybutton",
        "sphinx_design",
        "sphinxext.opengraph",
        "hoverxref.extension",
    ]
    if e is not None and _maybe(e) is not None
]

# local extensions (docs/extensions/*.py)
extensions += [p.stem for p in (HERE / "extensions").glob("*.py")]

# OpenGraph (optional; safe defaults)
ogp_site_url = repository_url
ogp_image = None  # set to a real absolute image URL if you have one

# -- General config ----------------------------------------------------------

autosummary_generate = True # generate API
autodoc_member_order = "bysource"
bibtex_reference_style = "author_year"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}  # api doc options, change here, new added

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False  # change here
napoleon_use_param = False  # change here
napoleon_custom_sections = [("Params", "Parameters")]

todo_include_todos = False

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")

nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints",
                    "build",   # new add
                    "build/**",   # new add
                    "**/build/**",   # new add
                    ]

# extlinks (issue/pr shortcuts)
extlinks = {
    "issue": (f"{repository_url}/issues/%s", "#%s"),
    "pr": (f"{repository_url}/pull/%s", "#%s"),
    "ghuser": ("https://github.com/%s", "@%s"),
}

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "pyro": ("http://docs.pyro.ai/en/stable/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "ml_collections": ("https://ml-collections.readthedocs.io/en/latest/", None),
    "mudata": ("https://mudata.readthedocs.io/en/latest/", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
    "huggingface_hub": ("https://huggingface.co/docs/huggingface_hub/main/en", None),
    "sparse": ("https://sparse.pydata.org/en/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = project_name
html_logo = "_static/BRIDGE_logo.png"

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "use_edit_page_button": True,
    "logo_only": True,
    "show_toc_level": 1,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "path_to_docs": "docs/",
    "repository_branch": github_branch_for_edit,
}

pygments_style = "default"
html_static_path = ["_static"]
html_css_files = ["css/override.css"]
html_js_files = ["js/custom.js"]
html_show_sphinx = False

def setup(app):
    """App setup hook."""
    app.add_config_value(
        "recommonmark_config",
        {
            "auto_toc_tree_section": "Contents",
            "enable_auto_toc_tree": True,
            "enable_math": True,
            "enable_inline_math": False,
            "enable_eval_rst": True,
        },
        True,
    )

# -- linkcode: map documented objects -> GitHub source URL --------------------

def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT)).strip().decode()

# Prefer branch name; fallback to commit SHA; fallback to "main"
try:
    git_ref = _git("rev-parse", "--abbrev-ref", "HEAD")
    if git_ref == "HEAD":
        git_ref = _git("rev-parse", "HEAD")
except Exception:
    git_ref = "main"

def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Return a GitHub URL for the given Python object (used by sphinx.ext.linkcode)."""
    if domain != "py":
        return None

    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname or not fullname:
        return None

    try:
        module = importlib.import_module(modname)
        obj: Any = module
        for part in fullname.split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)

        if isinstance(obj, property):
            obj = inspect.unwrap(obj.fget)  # type: ignore[arg-type]

        filename = inspect.getsourcefile(obj)
        if not filename:
            return None

        file_path = Path(filename).resolve()
        try:
            rel_path = file_path.relative_to(REPO_ROOT)
        except ValueError:
            return None

        src, lineno = inspect.getsourcelines(obj)
        end_line = lineno + len(src) - 1
        linespec = f"#L{lineno}-L{end_line}"

        return f"{repository_url}/blob/{git_ref}/{rel_path.as_posix()}{linespec}"
    except Exception:
        return None

# -- hoverxref ---------------------------------------------------------------

hoverx_default_type = "tooltip"
hoverxref_domains = ["py"]
hoverxref_role_types = dict.fromkeys(
    ["ref", "class", "func", "meth", "attr", "exc", "data", "mod"],
    "tooltip",
)

hoverxref_intersphinx = [
    "python",
    "numpy",
    "scanpy",
    "anndata",
    "lightning",
    "scipy",
    "pandas",
    "ml_collections",
    "ray",
]

if os.environ.get("READTHEDOCS"):
    hoverxref_api_host = "/_"
