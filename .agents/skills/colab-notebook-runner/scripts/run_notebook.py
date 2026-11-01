"""Headless Jupyter / Colab notebook executor using papermill and nbconvert."""

import argparse
import sys
import papermill as pm


def execute_notebook(input_path: str, output_path: str):
    print(f"Executing notebook: {input_path} -> {output_path}...")
    try:
        pm.execute_notebook(
            input_path,
            output_path,
            parameters={},
            kernel_name="python3",
            log_output=True,
        )
        print(f"Notebook execution succeeded: {output_path}")
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Jupyter notebook headlessly")
    parser.add_argument("--input", required=True, help="Input .ipynb path")
    parser.add_argument("--output", required=True, help="Output .ipynb path")
    args = parser.parse_args()

    execute_notebook(args.input, args.output)
