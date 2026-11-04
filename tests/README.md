# Automated Test Suite Guide

This folder contains automated safety and verification checks for MM-RFMNN.

Think of these tests like quality control inspectors at a car factory. Every time we change any code, we run these tests to prove that every single piece of math, every box, and every decision rule works exactly as promised.

---

## How to Run the Tests

From the main project folder, run:
```bash
pytest
```
All 27 automated tests run in about 2 to 3 seconds.

---

## What the Tests Actually Check in Plain English

### 1. Core Math & Box Geometry (`test_mm_rfmnn.py`)
- **Box Containment**: If a point is inside a box, does it get an exact score of 100% (1.0)? Yes.
- **Campfire Distance Falloff**: If a point is outside the box, does its score smoothly fade away as distance increases? Yes.
- **Rubber-Band Expansion**: When new samples arrive, does the box stretch to enclose them without growing larger than allowed? Yes.
- **Trimming Overlaps**: If an apple box and an orange box touch, does the code trim their edges so they never get confused? Yes.
- **Safe Scaling**: Does the normalizer scale numbers between 0 and 1 using only training data, without accidentally peeking at test answers? Yes.
- **Unplugged Sensors**: If we tell the model a sensor is disconnected, does it turn that sensor's vote to 0 without crashing or inventing fake numbers? Yes.
- **End-to-End Prediction**: Does the entire pipeline take data, learn patterns, and output valid probabilities that add up to 100%? Yes.
- **Clear Explanations**: Does the `.explain()` function clearly output the winning box coordinates and sensor weights for a decision? Yes.
- **Outlier Refusal**: If we feed the model extreme, crazy noise that looks nothing like the training data, does it safely refuse to guess (output `-1`)? Yes.

### 2. Stream Learning & Box Cleaning (`test_next_stage.py`)
- **Box Pruning**: Does the pruner find redundant or useless boxes and clean them out while keeping accuracy at 100%? Yes.
- **Streaming Ingestion**: Can the model ingest fresh batches of data on the fly (`partial_fit`) without having to start over from scratch? Yes.

### 3. Turnkey Encoders & Safety Guards (`test_phase4.py`)
- **Raw Media Encoders**: Can the system take raw 2D image pixels, raw 1D audio soundwaves, raw text sentences, and messy tables and translate all of them into clean unit coordinates? Yes.
- **Automatic Box Sizing**: Does `AdaptiveThetaEstimator` calculate the right box size automatically from sample spacing? Yes.
- **Contradiction Blocker**: If the vision sense says "Dog" with high confidence, but the audio sense says "Siren" with high confidence, does the system catch the argument and safely abstain? Yes.

### 4. Dataset Loaders & Caching (`test_dataset_loaders.py`)
- **Reliable Data Loading**: Verifies that all 5 benchmark datasets (UCI digits, Breast Cancer, Dermatology, Wearable activity, and Raw media) load with correct shapes, valid labels, and clean offline caching.
