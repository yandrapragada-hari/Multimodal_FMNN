# MM-RFMNN: Unit & Integration Test Suite

This directory contains automated unit and integration tests verifying the mathematical and architectural integrity of the MM-RFMNN framework.

## Test Runner Execution

To run all tests using `pytest`:
```bash
pytest
```

To run using Python's built-in `unittest`:
```bash
python -m unittest discover tests
```

---

## Test Coverage Overview

The test suite in [`test_mm_rfmnn.py`](./test_mm_rfmnn.py) includes 7 targeted verification suites:

1. **`test_hyperbox_membership`**:
   - Asserts that an input located inside a hyperbox ($V \le z \le W$) produces an exact membership score of $1.0$.
   - Asserts that an input outside the box decays exponentially with Manhattan distance: $\mu(z) = \exp(-\gamma D(z))$.

2. **`test_fmnn_expansion_and_fit`**:
   - Asserts that hyperboxes expand when new training samples fall within size threshold $\theta$.
   - Asserts that when expansion would exceed $\theta$, a new hyperbox is initialized.
   - Verifies overlap detection and Simpson's dimensional contraction between competing classes.

3. **`test_normalizer`**:
   - Verifies that `ModalityNormalizer` properly scales training data to $[0, 1]$ and safely handles constant-feature zero division.

4. **`test_missingness_masking`**:
   - Verifies that when an indicator mask $q_i^{(0)} = 0$ is provided, the fusion weight $\alpha_i^{(0)}$ becomes strictly $0.0$, and the remaining present modalities receive normalized simplex weights without zero-imputation hallucinations.

5. **`test_end_to_end_classifier`**:
   - Verifies complete `.fit()`, `.predict()`, and `.predict_proba()` workflow on a multimodal toy dataset.
   - Asserts probability simplex constraints ($\sum_c p(c) = 1.0$) and high accuracy retention.

6. **`test_explainability_payload`**:
   - Asserts that `.explain()` outputs winning hyperbox indices per class, modality reliability values, fusion weights, and uncertainty metrics for transparent geometric auditing.

7. **`test_abstention_on_extreme_outlier`**:
   - Verifies that an extreme out-of-distribution input triggers the multi-criteria abstention policy, emitting class label `-1`.
