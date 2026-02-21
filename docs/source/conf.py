# Sphinx configuration for lrnnx
# Using Furo theme with native navbar icon support

import os
import sys
import re
# Add project root to Python path
sys.path.insert(0, os.path.abspath('../..'))

# -- Mock all external dependencies --
autodoc_mock_imports = [
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.utils',
    'torch.utils.checkpoint', 'torch.cuda', 'torch.autograd',
    'torch.distributed', 'numpy', 'scipy', 'einops',
    'causal_conv1d', 'mamba_ssm', 'selective_scan_cuda',
    'triton', 'triton.language', 'opt_einsum', 'pscan',
    'flash_attn', 'cuda', 'cupy','simplified_scan_cuda'
]

# -- Project information --
project = 'lrnnx'
copyright = '2026, SforAiDl'
author = 'Karan Bania, Soham Kalburgi, Manit Tanwar, Dhruthi, Aditya Nagarsekar, Harshvardhan Mestha, Naman Chibber, Raj Deshmukh, Anish Sathyanarayanan, Aarush Rathore, Pratham Chheda'
version = '1.0.0'
release = '1.0.0'

# -- General configuration --
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

myst_enable_extensions = ["colon_fence"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output options --
html_theme = 'furo'  # Modern theme with icon support

html_theme_options = {
    # Add icons to top-right navbar
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/SforAiDl/lrnnx/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    
    # Add icon links in header
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/SforAiDl/lrnnx",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/lrnnx/",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 24 24">
                    <path d="M12 2L2 7v10l10 5 10-5V7L12 2zm0 2.18l7.16 3.59L12 11.35 4.84 7.77 12 4.18zM4 9.47l7 3.5v6.86l-7-3.5V9.47zm9 10.36v-6.86l7-3.5v6.86l-7 3.5z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# -- Autodoc options --
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
}

# -- Napoleon options --
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Intersphinx mapping --
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Remove module names for cleaner display
add_module_names = False

def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    """Remove *args and **kwargs (and variants like **mixer_kwargs) from autodoc signatures."""
    if signature:
        # Remove **kwargs, **mixer_kwargs, **layer_args, etc. (with optional type annotation)
        signature = re.sub(r',?\s*\*\*\w+', '', signature)
        # Remove *args (with optional type annotation)
        signature = re.sub(r',?\s*\*args\b[^,)]*', '', signature)
        # Clean up leftover formatting
        signature = re.sub(r'\(\s*,', '(', signature)
        signature = re.sub(r',\s*\)', ')', signature)
        # Resolve np alias so intersphinx can link numpy.ndarray
        signature = re.sub(r'\bnp\.ndarray\b', 'numpy.ndarray', signature)
    if return_annotation:
        return_annotation = re.sub(r'\bnp\.ndarray\b', 'numpy.ndarray', return_annotation)
    return signature, return_annotation


def autodoc_process_docstring(app, what, name, obj, options, lines):
    """Remove *args/**kwargs entries from docstring parameter lists (all formats)."""
    indices_to_remove = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Pattern for RST-style directives (with or without * escaping):
        #   :param \*\*kwargs: ...  /  :param \*\*mixer_kwargs: ...  /  :param \*\*layer_args: ...
        #   :param kwargs: ...  /  :param mixer_kwargs: ...
        #   :type \*\*kwargs: ...  /  :type kwargs: ...
        #   :keyword kwargs: ...
        is_param_line = bool(re.match(
            r'^:(param|type|keyword)\s+(\\?\*\\?\*\s*\w+|(\w*kwargs|args))\s*:', stripped
        ))

        # Pattern for Google/NumPy style:
        #   **kwargs: ...  /  *args: ...  /  **mixer_kwargs: ...  /  **layer_args: ...
        is_raw_line = bool(re.match(
            r'^\*{1,2}\w+\s*[\(:]', stripped
        ))

        if is_param_line or is_raw_line:
            indices_to_remove.append(i)
            # Also remove continuation lines (indented, no new directive)
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if not next_line.strip():
                    break
                if re.match(r'^:', next_line.strip()) or not next_line[0:1].isspace():
                    break
                indices_to_remove.append(j)
                j += 1
            i = j
        else:
            i += 1

    for idx in reversed(indices_to_remove):
        del lines[idx]


def setup(app):
    app.connect('autodoc-process-signature', autodoc_process_signature)
    app.connect('autodoc-process-docstring', autodoc_process_docstring)