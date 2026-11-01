"""Generates Google Colab compatible .ipynb notebook from python scripts and markdown docs."""

import argparse
import json
import os
import nbformat as nbf


def create_colab_notebook(script_path: str, output_path: str, title: str):
    nb = nbf.v4.new_notebook()

    # Title & Badge Cell
    header_md = f"""# {title}
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

This notebook was automatically generated with complete environment setup, reproducible random seeds, and modular architecture.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(header_md))

    # Setup & Dependency Cell
    setup_code = """# Verify environment & dependencies
import sys, os
try:
    import numpy as np, scipy, sklearn
except ImportError:
    !pip install -q scikit-learn numpy scipy pandas
    import numpy as np, scipy, sklearn

print(f"NumPy: {np.__version__} | Scikit-Learn: {sklearn.__version__}")
"""
    nb.cells.append(nbf.v4.new_code_cell(setup_code))

    # Main Script Code
    with open(script_path, "r", encoding="utf-8") as f:
        script_content = f.read()

    # In interactive notebooks, __file__ does not exist; replace with os.getcwd()
    script_content = script_content.replace("os.path.dirname(__file__)", "os.getcwd()")

    nb.cells.append(nbf.v4.new_markdown_cell("## Implementation & Benchmark Execution"))
    nb.cells.append(nbf.v4.new_code_cell(script_content))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully generated Colab notebook: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Python script to Google Colab Notebook")
    parser.add_argument("--script", required=True, help="Input python script path")
    parser.add_argument("--output", required=True, help="Output .ipynb path")
    parser.add_argument("--title", default="Google Colab Notebook", help="Notebook title")
    args = parser.parse_args()

    create_colab_notebook(args.script, args.output, args.title)
