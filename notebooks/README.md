# MM-RFMNN: Interactive Notebooks (Phase 2)

This directory contains interactive Jupyter and Google Colab notebooks for demonstration, experimentation, and reproducibility.

## Notebook Inventory

| File | Description | Status |
| :--- | :--- | :--- |
| [`MM_RFMNN_Colab_Benchmark.ipynb`](./MM_RFMNN_Colab_Benchmark.ipynb) | Complete Google Colab ready notebook featuring "Open in Colab" badge, automatic environment setup, synthetic dataset generation, and benchmark execution. | Ready for Upload / Interactive Run |
| [`MM_RFMNN_Executed.ipynb`](./MM_RFMNN_Executed.ipynb) | Full executed notebook containing saved cell outputs, logs, and empirical validation results across all 4 research questions. | Executed & Verified via Papermill |

---

## Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `MM_RFMNN_Colab_Benchmark.ipynb`.
3. Run all cells (`Runtime -> Run all`).

---

## Running Locally

To re-run the notebook headlessly from the project root using `papermill`:
```bash
python .agents/skills/colab-notebook-runner/scripts/run_notebook.py \
  --input "notebooks/MM_RFMNN_Colab_Benchmark.ipynb" \
  --output "notebooks/MM_RFMNN_Executed.ipynb"
```
Or start a local Jupyter server:
```bash
jupyter notebook notebooks/
```
