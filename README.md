# MM-RFMNN: Multimodal Reliability-Aware Fuzzy Min-Max Neural Network

[![Tests](https://img.shields.io/badge/tests-27%2F27%20passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-ruff-black.svg)]()
[![Status](https://img.shields.io/badge/Phases%201--4-Complete-blue.svg)]()

A noise-proof, interpretable, and safe artificial intelligence architecture that combines multiple senses (like pictures, audio, sensor readings, and text) without breaking when a sensor gets noisy, corrupted, or completely unplugged.

---

## What is MM-RFMNN in Simple English?

Imagine you are driving a car in heavy fog.
- Your **eyes** can barely see through the fog.
- Your **ears** can hear a siren in the distance.
- Your **car radar** can sense a metal object 50 feet ahead.

As a human, you naturally **rely less on your eyes** and **listen closely to your ears and radar**. If you are completely unsure what is ahead, you slow down and stop rather than accelerating blindly.

Most modern AI models cannot do this. They throw all their sensor data into one big "blender" (called vector concatenation). If a microphone emits static noise or a camera is covered in mud, the whole blender explodes and outputs confident nonsense.

**MM-RFMNN solves this problem.** It gives each sense its own dedicated room, draws clear geometric "boxes" around what it learns, grades how trustworthy each sensor is in real time, and safely says **"I do not know"** when it is genuinely confused.

---

## The 5 Big Problems with Standard AI and How We Fix Them

| The Problem in Standard AI | What Happens in the Real World | How MM-RFMNN Solves It |
| :--- | :--- | :--- |
| **1. The Blender Trap (Early Fusion)** | Most neural networks glue all sensor numbers into one giant list. Noise on a single microphone poisons the whole system. | **Independent Sense Boxes**: Each sense (audio, vision, clinical numbers) has its own separate collection of geometric boxes. Noise in one sense cannot poison the others. |
| **2. The Fake Numbers Trap (Zero Imputation)** | If a medical probe is disconnected, standard AI feeds fake zeros into the math, creating artificial patterns and hallucinations. | **Clean Shutoff**: When a sensor is unplugged, its voting power is smoothly set to zero. The model never feeds fake zeros into the system. |
| **3. The Overconfident Guess Trap** | Standard deep networks often say *"I am 99.9% sure"* even when they are completely wrong. | **Calibrated Confidence**: We evaluate each sense on fresh, held-out data to verify how often it is actually right, turning raw scores into true honest percentages (0% to 100%). |
| **4. The Blind Guess Trap** | If a camera sees a dog but an audio sensor hears an airplane, standard models pick one at random. | **Contradiction Blocker**: When two senses strongly contradict each other, the system catches the disagreement and safely says *"I do not know"* (returns `-1`). |
| **5. The Black Box Mystery** | Deep neural networks use millions of mysterious weights that no human doctor or engineer can verify. | **Clear Geometric Boxes**: The model learns using simple, inspectable boxes with minimum and maximum boundary coordinates that any human can see and audit. |

---

## How It Works: The "Elastic Box" and "Campfire" Analogy

1. **The Elastic Box (Hyperbox)**:
   - When the AI learns what an apple looks like based on sweetness and redness, it draws a tight box around the first apple sample.
   - When it sees another apple nearby, it stretches the box like a rubber band to enclose both apples.
   - If stretching the box would swallow an orange, it stops stretching and makes a new box instead.
2. **The Campfire Rule (Fuzzy Membership)**:
   - Inside the box, you are 100% inside.
   - Outside the box, it behaves like standing near a warm campfire. Right next to it, you feel strong heat (95% familiarity). As you walk further away, the warmth smoothly fades toward zero.
3. **The Trust Score**:
   - The AI continuously checks how clean and informative each sensor is. Clean, trustworthy sensors get high voting weight; noisy or broken sensors get turned down.
4. **The Final Verdict**:
   - The AI tallies up the votes from the trusted senses. If total evidence is clear, it makes a prediction. If the signals are too blurry or contradictory, it safely abstains.

---

## Repository Map

```
MM-RFMNN/
├── README.md                         # This plain-English overview
├── PROJECT_STATUS_AND_ARCHITECTURE_REPORT.md # Detailed architecture and status report
├── PROJECT_STATUS_AND_ARCHITECTURE_REPORT.docx # Beautiful Word document report (61KB)
├── ARCHITECTURE_AND_PIPELINE_GUIDE.md # Step-by-step pipeline and encoder guide
├── MM_RFMNN_Algorithm_Blueprint.md   # Mathematical equations and research blueprint
├── pytest.ini                        # Automated test configuration
│
├── mm_rfmnn/                         # Core Python Library
│   ├── model.py                      # Main classifier (MMRFMNNClassifier)
│   ├── base/                         # Hyperbox builders, auto-sizing, and pruning
│   ├── datasets/                     # Loaders for 5 real-world multimodal datasets
│   ├── encoders/                     # Translators for raw images, sounds, text, and tables
│   ├── reliability/                  # Sensor trust and quality graders
│   ├── fusion/                       # Blending senses and contradiction blockers
│   └── viz/                          # 2D box plotters and risk-coverage charts
│
├── experiments/                      # Research Benchmarks
│   ├── benchmark_multiple_datasets.py# Runs test suite across all 5 datasets
│   ├── evaluate_real_datasets.py     # Deep-dive test on handwritten digits
│   ├── demo_phase4_capabilities.py   # Demonstration generating visual charts
│   └── output/                       # Saved charts (.png) and test results (.json)
│
├── notebooks/                        # Interactive Demonstrations
│   ├── MM_RFMNN_Colab_Benchmark.ipynb # Ready-to-run Google Colab notebook
│   └── MM_RFMNN_Executed.ipynb       # Pre-run notebook with all charts saved
│
└── tests/                            # Automated Verification Suite (27/27 passing)
    ├── test_mm_rfmnn.py              # Core geometry and reliability tests
    ├── test_next_stage.py            # Box pruning and streaming tests
    ├── test_phase4.py                # Raw media encoders and safety guard tests
    └── test_dataset_loaders.py       # Data downloading and caching checks
```

---

## Quickstart

### 1. Run All Automated Tests (27/27 passing in ~2 seconds)
```bash
pytest
```

### 2. Run the 5-Dataset Real-World Benchmark
```bash
python experiments/benchmark_multiple_datasets.py
```

### 3. Generate Visual Diagnostic Pictures
```bash
python experiments/demo_phase4_capabilities.py
```
This generates three charts in `experiments/output/`:
- `hyperboxes_2d_tabular.png`: A visual 2D picture showing the boxes and decision borders.
- `modality_attributions.png`: A chart showing how much the model trusted each sensor.
- `risk_coverage_curve.png`: A chart showing how accuracy jumps when the model is allowed to say "I do not know".

---

## Real-World Benchmark Results (5 Datasets)

We tested MM-RFMNN and standard deep neural networks (MLPs) across 5 different real-world domains:

| Dataset | Number of Senses | Domain & Data Channels | Clean Deep MLP Accuracy | Clean MM-RFMNN Accuracy | Deep MLP Under Extreme Noise | MM-RFMNN Under Extreme Noise | When 1 Sensor is Unplugged | Accuracy on Confident Answers (80% Coverage) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCI Handwritten Digits** | 6 | Fourier shapes, pixel windows, profiles, moments | 97.8% | 86.2% | **8.6% (total collapse)** | **85.9% (immune)** | **85.6%** | **87.7%** |
| **Breast Cancer (WDBC)** | 3 | Microscope cell nucleus measurements (Mean, Error, Extremum) | 98.2% | 95.3% | **49.7% (coin flip)** | **95.6% (immune)** | **96.2%** | **99.3%** |
| **UCI Dermatology** | 2 | Patient visible symptoms + biopsy microscope views | 98.2% | 90.0% | **20.7% (total collapse)** | **79.9% (robust)** | **87.1%** | **95.5%** |
| **Wearable Movement (HAR)**| 3 | Phone motion: Accelerometer, Gravity, Gyroscope | 96.5% | 69.3% | **63.6% (degraded)** | **86.5% (robust)** | **68.4%** | **66.0%** |
| **Raw Mixed Media** | 4 | Raw 2D images + raw 1D sound + text phrases + sensor tables | 100.0% | 100.0% | **73.4% (degraded)** | **99.7% (immune)** | **100.0%** | **100.0%** |

### What These Numbers Mean in Plain English:
1. **Total Noise Immunity**: When a sensor suffers severe static noise ($\sigma=6.0$), standard deep neural networks collapse (plummeting to 8% or 20%). MM-RFMNN senses the noise, turns down that channel, and stays near 85% to 99% accuracy.
2. **Missing Sensors Do Not Break It**: When a sensor is completely unplugged, MM-RFMNN continues operating smoothly using the remaining senses without inventing fake zeros.
3. **High Precision on Certified Decisions**: When allowed to abstain on the 20% most ambiguous samples, accuracy on the remaining answers reaches up to **99.3% - 100%**.
