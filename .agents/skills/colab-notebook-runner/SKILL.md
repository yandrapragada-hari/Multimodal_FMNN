---
name: colab-notebook-runner
description: >-
  Creates, formats, converts, and executes Jupyter / Google Colab notebooks (.ipynb)
  headlessly from command-line scripts or models, verifying end-to-end execution.
---

# Google Colab & Jupyter Notebook Runner Skill

## Capabilities
- Converts stand-alone Python algorithms and experiment scripts into interactive, runnable Google Colab notebooks with markdown explanations and "Open in Colab" badge.
- Executes Jupyter notebooks headlessly using `papermill` and `nbconvert`, validating output cells and error-free execution.
- Extracts execution metrics, plots, and outputs directly from `.ipynb` files.

## Available Helpers
- `scripts/py_to_colab.py`: Converts Python script + markdown blueprint into an executable Google Colab `.ipynb`.
- `scripts/run_notebook.py`: Headlessly executes a notebook and saves the executed notebook with all outputs.

## Usage Guide

### 1. Generating a Colab Notebook
```bash
python ~/.gemini/config/skills/colab-notebook-runner/scripts/py_to_colab.py --title "MM-RFMNN: Multimodal Reliable Classification" --script "experiments/benchmark.py" --output "notebooks/MM_RFMNN_Colab.ipynb"
```

### 2. Headless Notebook Execution & Verification
```bash
python ~/.gemini/config/skills/colab-notebook-runner/scripts/run_notebook.py --input "notebooks/MM_RFMNN_Colab.ipynb" --output "notebooks/MM_RFMNN_Executed.ipynb"
```
