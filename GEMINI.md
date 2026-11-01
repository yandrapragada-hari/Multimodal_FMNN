# Project Standards & Anti-Slop Directives

## 1. Technical Rigor & Code Integrity
- **No Filler / Placeholder Slop**: Every script and function must be fully realized and self-contained. Never leave `# TODO: implement` or `# Logic goes here`.
- **High-Performance Computation**: Always prefer vectorized NumPy / SciPy operations over native Python loops for mathematical operations.
- **Strict Typing**: All classes and function signatures must include comprehensive type hints (`typing.Optional`, `typing.List`, `typing.Tuple`, `typing.Dict`, `typing.Any`).

## 2. Machine Learning Best Practices
- **Leakage Prevention**: All scalers, normalizers, and dimensional bounds must be fitted strictly on `X_train`. Never fit scalers on the entire dataset prior to splitting.
- **Unbiased Meta-Targets**: Ground-truth correctness targets for reliability estimation must be generated via out-of-fold cross-validation (`fit_oof`), never in-sample.
- **Explicit Missingness**: Handle missing modalities using binary presence masks ($q^{(k)}$). Never pass imputed zero-vectors through distance-based hyperboxes.
- **Evidence Calibration**: Always apply temperature calibration before fusing modality class evidence.

## 3. Testing & Verification
- Verify all implementations with automated tests (`pytest` or `unittest`).
- Run `ruff check` on modified Python source files before concluding a task.
