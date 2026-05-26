# Logistic Regression from Scratch

A small machine learning project that implements binary logistic regression in NumPy and applies it to university admission prediction from two exam scores.

## Overview

This repository walks through the full pipeline for a classic classification problem:

- Load and visualize a 2D dataset
- Implement the sigmoid function, cost function, and gradient
- Train with batch gradient descent
- Plot the learned decision boundary
- Evaluate predictions on held-out-style examples

The implementation uses only NumPy for the core model; Matplotlib is used for visualization and SciPy for optional gradient checking.

## Dataset

`data/data.txt` contains 100 labeled examples. Each row has:

| Column | Description |
|--------|-------------|
| 1 | Exam 1 score |
| 2 | Exam 2 score |
| 3 | Admission label (`1` = admitted, `0` = not admitted) |

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook notebooks/logistic_regression.ipynb
```

Run all cells from the project root (or ensure paths resolve relative to the notebook).

## Project structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── data.txt
└── notebooks/
    └── logistic_regression.ipynb
```

## Results

Training learns a linear decision boundary in feature space. Example predictions (high scores → admitted, low scores → not admitted) are printed at the end of the notebook, along with training accuracy.

## Author

Jordan Kim
