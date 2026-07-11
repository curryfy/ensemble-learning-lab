# Ensemble Learning Lab

This is a simple educational Python project for learning Chapter 16, Ensemble Learning, from *The Elements of Statistical Learning* through hands-on scikit-learn experiments.

The project starts with a small breast cancer sanity-check baseline, then moves to MNIST for voting and stacking ensemble experiments.

## Project stages

1. Stage 1: Breast Cancer Sanity-Check Baseline
   - Load the scikit-learn breast cancer dataset.
   - Build simple metrics and plotting helpers.
   - Compare a first table of baseline model results.
   - Keep this stage unchanged as a quick binary-classification check.

2. Stage 2: MNIST Voting Ensemble
   - Use MNIST as the main ensemble-learning exercise.
   - Compare Random Forest, Extra-Trees, an SVM-like linear classifier, and Logistic Regression.
   - Build and compare hard voting and soft voting ensembles.
   - Save a results table and model-comparison figure when the notebook is run.

3. Stage 3: Stacking / Blender
   - Use the Stage 2 base learners to create meta-features from MNIST predictions.
   - Train a Logistic Regression blender on those meta-features.
   - Compare stacking against Extra-Trees, Random Forest, hard voting, and soft voting.
   - Inspect blender coefficients to interpret each base learner's contribution.

## Why MNIST for Stage 2?

MNIST is more suitable for studying ensemble learning than the breast cancer dataset because it is larger, multi-class, and higher-dimensional. The breast cancer dataset is still useful as a quick sanity check, but MNIST gives the individual classifiers more room to make different kinds of mistakes. That makes it a better setting for studying when voting ensembles help.

Stage 2 compares individual classifiers against hard voting and soft voting:

- Random Forest
- Extra-Trees
- SVM-like linear classifier
- Logistic Regression
- Hard voting ensemble
- Soft voting ensemble

## Stage 3 stacking/blender

Stage 3 extends the MNIST experiment from equal-weight voting to a learned combination rule. The base learners are the same model families used in Stage 2: Extra-Trees, Random Forest, an SVM-like linear classifier, and Logistic Regression.

Each fitted base learner produces predictions on the validation and test sets. These outputs become meta-features for a second-level model. The blender is a simple Logistic Regression model trained on validation-set meta-features, so the experiment asks whether learned combination weights can improve on hard voting and soft voting.

The Stage 3 notebook compares stacking with Extra-Trees, Random Forest, hard voting, and soft voting. It also includes a blender coefficient analysis: larger absolute coefficients suggest that the blender gives a base learner more influence in its final predictions.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the notebooks

Stage 1 breast cancer baseline:

```bash
jupyter notebook notebooks/01_baselines_breast_cancer.ipynb
```

Stage 2 MNIST voting ensemble:

```bash
jupyter notebook notebooks/02_mnist_voting_ensemble.ipynb
```

The Stage 2 notebook uses a fast mode by default: 20,000 training examples, 5,000 validation examples, and 10,000 test examples. Set `FAST_MODE = False` in the notebook to use the full book-style split of 50,000 train, 10,000 validation, and 10,000 test examples.

Stage 3 MNIST stacking/blender:

```bash
jupyter notebook notebooks/03_mnist_stacking_blender.ipynb
```
