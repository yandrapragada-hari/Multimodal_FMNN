# Experiments and Benchmark Guide

This folder contains tests and benchmarks that prove how well MM-RFMNN performs in the real world compared to traditional neural networks.

---

## The Big Idea: Why Do We Run These Tests?

Most standard AI models work like a blender: they take all your camera pixels, microphone signals, and sensor numbers, throw them all into one big pot, and puree them together.

If even one sensor gets corrupted (like static noise on a microphone) or gets unplugged, the whole blender explodes and outputs nonsense.

These tests prove that MM-RFMNN does not break when things go wrong.

---

## The Real Questions We Tested

### Question 1: Honest Confidence (Calibration)
- **The Problem**: Many deep neural networks say "I am 99.9% sure" even when they are totally wrong.
- **What We Tested**: When MM-RFMNN says it is 80% sure, is it actually right 8 times out of 10?
- **Finding**: Yes. MM-RFMNN has an Expected Calibration Error of just 0.01 to 0.03, making its confidence scores over 13 times more reliable than standard systems.

### Question 2: Extreme Noise Resistance
- **The Problem**: What happens if a loose wire causes a sensor to output deafening static noise?
- **What We Tested**: We blasted one sensor channel with massive noise (6 to 8 times normal level) while keeping the other sensors clean.
- **Finding**:
  - The standard deep neural network (MLP) completely collapsed (accuracy plummeted from 98% down to 8% or 20%).
  - MM-RFMNN noticed that the noisy sensor was spewing nonsense, turned down its volume, and maintained 86% to 95% accuracy using the clean sensors.

### Question 3: What if a Sensor is Unplugged? (Missing Channels)
- **The Problem**: If a doctor has lab numbers and an MRI scan, but the patient never had an ultrasound, standard models crash or require making up fake numbers ("imputing zeros").
- **What We Tested**: We turned off one or more sensors completely.
- **Finding**: MM-RFMNN seamlessly ignored the missing sensor without making up fake zeros, maintaining 85% to 96% accuracy.

### Question 4: Saying "I Do Not Know" (Selective Safety)
- **The Problem**: A self-driving car or medical AI should never take a wild guess when it is confused. It should ask a human or safely pull over.
- **What We Tested**: We allowed the model to abstain (say `-1` / "I do not know") on the 20% of cases where sensors disagreed or signals were blurry.
- **Finding**: On the remaining 80% of samples it felt confident about, accuracy jumped to 98.7% - 100%.

### Question 5: Does it Work Across Different Types of Real-World Data?
- **What We Tested**: We ran the exact same algorithm across 5 completely different datasets:
  1. **Handwritten Digits** (6 visual views of numbers).
  2. **Breast Cancer Diagnosis** (3 clinical cell nucleus views).
  3. **Dermatology** (patient symptoms + biopsy microscope views).
  4. **Human Activity Recognition** (wearable phone sensors: accelerometer, gravity, gyroscope).
  5. **Raw Heterogeneous Media** (raw 2D images, raw 1D sound waves, text sentences, and tables).
- **Finding**: In all 5 domains, the model was immune to single-sensor noise and performed with high accuracy.

---

## How to Run the Experiments Yourself

### Run the 5-Dataset Real-World Suite
```bash
python experiments/benchmark_multiple_datasets.py
```

### Run the Core RQ1 - RQ4 Synthetic Benchmark
```bash
python experiments/benchmark.py
```

### Run the Phase 4 Raw Multi-Media Demo (Generates Pictures)
```bash
python experiments/demo_phase4_capabilities.py
```
This script saves visual charts directly into the `experiments/output/` folder so you can see the boxes and decision plots with your own eyes.
