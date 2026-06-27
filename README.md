# Ensemble Learning Lab

This is a simple educational Python project for learning Chapter 16, Ensemble Learning, from *The Elements of Statistical Learning* through hands-on scikit-learn experiments.

The lab starts with the built-in breast cancer dataset from scikit-learn so the first examples can focus on model behavior instead of data collection.

## Project stages

1. Stage 1: Ensemble Baselines
   - Load a binary classification dataset.
   - Build simple metrics and plotting helpers.
   - Prepare notebooks for baseline model experiments.

2. Stage 2: ISLE / Lasso Post-processing
   - Explore post-processing many basis learners with sparse linear models.
   - Keep this stage for later work.

3. Stage 3: Simplified Rule Ensemble / RuleFit
   - Explore a beginner-friendly version of rule-based ensemble ideas.
   - Keep this stage for later work.

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

## Start the notebook

```bash
jupyter notebook notebooks/01_baselines_breast_cancer.ipynb
```

Stage 1 currently loads the breast cancer dataset, checks train/test shapes, and reviews class balance before baseline models are added.
