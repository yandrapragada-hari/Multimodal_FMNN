# MM-RFMNN: Core Code Engine Guide

This folder contains the complete code for the **Multimodal Reliability-Aware Fuzzy Min-Max Neural Network (MM-RFMNN)**.

In simple terms, this is an AI system that takes inputs from multiple senses (such as pictures, sound, text, and numbers), draws neat geometric boxes around what it learns, grades how much it can trust each sense, and makes safe, explainable decisions.

---

## Folder Layout

```
mm_rfmnn/
├── __init__.py               # Main library exports
├── model.py                  # The main controller (MMRFMNNClassifier)
│
├── base/                     # The core geometric box builders
│   ├── hyperbox.py           # Creates, expands, and shrinks boxes around data points
│   ├── adaptive_theta.py     # Automatically calculates how big boxes should be
│   ├── pruning.py            # Removes redundant or useless boxes to save memory
│   └── incremental.py        # Lets the model learn on the fly from live data streams
│
├── datasets/                 # Real-world data loaders
│   └── loader.py             # Loads and caches datasets (digits, medical, wearables)
│
├── encoders/                 # Data cleaners and translators
│   ├── normalizer.py         # Scales numbers cleanly between 0 and 1
│   └── multimodal.py         # Turnkey converters for raw images, sounds, text, and tables
│
├── reliability/              # The trust and quality checker
│   └── estimator.py          # Figures out which sensors are giving good information
│
├── fusion/                   # Combining senses safely
│   ├── fuser.py              # Blends trusted senses together into one final probability
│   └── conflict.py           # Stops the model if two senses strongly argue with each other
│
└── viz/                      # Visual explanations
    └── plotter.py            # Draws clean 2D pictures of the boxes and decision charts
```

---

## What Each File Does in Plain English

### 1. `model.py` (The Main Remote Control)
- **Class**: `MMRFMNNClassifier`
- **What it does**: This is the single tool you interact with. It works just like standard scikit-learn models.
- **Key Actions**:
  - `fit(X, y)`: Trains the network on your data.
  - `predict(X)`: Tells you what class a new sample belongs to (or returns `-1` if it is too unsure to guess).
  - `predict_proba(X)`: Gives you true percentage chances (e.g., 92% Class A, 8% Class B).
  - `explain(X)`: Gives you an audit report explaining exactly which boxes matched, which sensors were trusted, and why the decision was made.

### 2. `base/hyperbox.py` (The Rubber-Band Box Builder)
- **Classes**: `HyperboxBank`, `FMNN`
- **What it does**:
  - Imagine scattering red dots and blue dots on a table.
  - When the AI sees a red dot, it draws a tiny box around it.
  - When it sees another red dot nearby, it stretches the box like a rubber band to enclose both dots.
  - If stretching would make the box too big, or if it would swallow a blue dot, it stops and makes a new box instead.
  - If two boxes of different colors slightly touch, it trims their edges so they never overlap.
- **Campfire Analogy**: Inside a box, you are 100% in. Outside a box, warmth fades smoothly as you walk further away. This is called "fuzzy membership."

### 3. `base/adaptive_theta.py` (Automatic Box Sizing)
- **Class**: `AdaptiveThetaEstimator`
- **What it does**: In older systems, humans had to guess how big the boxes should be. If you guess too small, you get a million tiny boxes. If you guess too big, different classes get lumped together.
- **How it solves this**: It looks at the spacing between samples of the same class and automatically chooses the right box size for each sense.

### 4. `base/pruning.py` (The Box Cleaner)
- **Class**: `HyperboxPruner`
- **What it does**: Over time, some boxes only ever catch one random noise point, or a small box gets completely swallowed inside a larger box of the same class.
- **Benefit**: This file cleans out those useless and redundant boxes. On real data, it cut box count by 66% while keeping accuracy at 100% of its original level.

### 5. `base/incremental.py` (Continuous Learning)
- **Class**: `OnlineStreamAdapter`
- **What it does**: Lets the AI learn from continuous streams of data one batch at a time (`partial_fit`) without having to start over and re-train from scratch.

### 6. `encoders/normalizer.py` and `encoders/multimodal.py` (Translators)
- **Classes**: `ModalityNormalizer`, `ImageFeatureEncoder`, `AudioFeatureEncoder`, `TextFeatureEncoder`, `TabularQuantileEncoder`
- **What it does**: Different senses speak different languages. Images have pixels, audio has frequencies, sensors have voltages. This code translates all of them into neat numbers between 0.0 and 1.0 so the boxes can wrap around them evenly.

### 7. `reliability/estimator.py` (The Trust Judge)
- **Classes**: `ReliabilityEstimator`, `MetaFeatureExtractor`
- **What it does**: Just because a sensor produces numbers does not mean those numbers are good. If a microphone is producing loud static or a camera is covered in mud, this module grades that sensor with low reliability (e.g. 0.05 out of 1.0).

### 8. `fusion/fuser.py` and `fusion/conflict.py` (The Decision Board)
- **Classes**: `ReliabilityFuser`, `TemperatureCalibrator`, `CrossModalConflictDetector`
- **What it does**:
  - Gathers the evidence from all senses.
  - Gives more voting power to reliable senses and mutes broken or noisy senses.
  - If a sensor is completely unplugged, it simply turns off that sensor's vote without crashing.
  - If the eyes say "Cat" with 99% certainty, but the ears say "Airplane" with 99% certainty, the conflict detector notices the contradiction and safely says "I do not know" instead of picking one blindly.

### 9. `viz/plotter.py` (Visual Diagrams)
- **Functions**: `plot_hyperboxes_2d`, `plot_modality_attributions`, `plot_risk_coverage_curve`
- **What it does**: Draws charts and pictures showing the boxes, the points, the sensor weights, and how accurate the model is when it only answers questions it feels confident about.
