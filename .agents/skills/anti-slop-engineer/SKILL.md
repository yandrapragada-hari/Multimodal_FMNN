---
name: anti-slop-engineer
description: >-
  Enforces zero-slop coding standards, mathematical rigor, strict type safety, vectorized
  performance, and elimination of generic AI boilerplate across all machine learning and software tasks.
---

# Anti-Slop Engineering Skill

## Purpose
This skill ensures all generated code, architectures, and documentation meet production and high-impact academic standards, eliminating filler text, hallucinated APIs, unnecessary abstraction layers, and unoptimized loops.

## Core Directives

### 1. Zero Boilerplate & High Signal-to-Noise
- Never generate superficial wrappers that merely call another function without adding domain logic.
- Avoid placeholder comments like `# Add code here` or `# Implement your logic here`. Every routine must be fully functional and self-contained.
- Do not repeat documentation in responses when it is already preserved in an artifact or markdown document.

### 2. High-Performance Vectorization
- Replace Python `for`-loops over datasets with vectorized NumPy / SciPy array operations and broadcasting.
- Pre-allocate memory buffers where sizes are known.
- Avoid copying large arrays repeatedly; use views or in-place operations (`out=...`) where safe.

### 3. Strict Type Safety & Contracts
- Add standard Python type hints (`from typing import Optional, List, Tuple, Dict, Any, Union`).
- Add defensive input assertions and shape checks at the boundaries of public APIs.
- Validate bounds (e.g. $[0, 1]$ intervals, non-empty arrays, probability distributions summing to 1).

### 4. Leakage-Free ML Methodology
- Strict separation of preprocessing fitting: normalizers, scalers, and encoders must ONLY be fit on training partitions.
- Any meta-model or reliability model predicting correctness MUST use Out-Of-Fold (OOF) cross-validation targets.
- Missing modalities must be explicitly masked rather than filled with arbitrary zeros that trigger distance anomalies.

### 5. Automated Verification
- Every module must be accompanied by unit tests covering nominal execution, boundary conditions, and failure modes.
- Format with `ruff` and verify type assertions before claiming completion.
