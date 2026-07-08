# Stage 2: MNIST Voting Ensemble Summary

## 1. Project Motivation

This project studies ensemble learning concepts from ESL Chapter 16 through hands-on scikit-learn experiments. Stage 1 uses the breast cancer dataset as a small sanity-check baseline. Stage 2 focuses on a larger question: whether naive voting ensembles improve MNIST classification compared with individual classifiers.

The goal is not to reach state-of-the-art MNIST performance. Instead, the goal is to make ensemble behavior visible in a controlled, readable experiment.

## 2. Dataset

MNIST is a 10-class handwritten digit classification task. Each observation is a grayscale digit image represented as flattened pixel features. The task is to classify each image as one of the digits 0 through 9.

Compared with the Stage 1 breast cancer dataset, MNIST is larger, multi-class, and higher-dimensional. This makes it a more useful setting for studying how different model families behave and whether voting improves predictive performance.

## 3. Models Compared

The Stage 2 experiment compares the following models:

- Extra-Trees: an ensemble of randomized decision trees. It adds randomness in tree construction and can capture nonlinear pixel interactions.
- Random Forest: an ensemble of decision trees trained with bootstrap sampling and random feature selection. It is a strong tree-based baseline for tabular-style feature vectors.
- Linear SVM-like SGD classifier: a fast linear classifier trained with stochastic gradient descent and hinge loss. It approximates a linear SVM-style decision boundary.
- Logistic Regression: a linear probabilistic classifier. It provides class probabilities, which are useful for soft voting.
- Hard Voting: an ensemble that predicts the class receiving the most votes from the fitted base classifiers.
- Soft Voting: an ensemble that averages predicted class probabilities from fitted classifiers that support `predict_proba`.

## 4. Evaluation Metrics

The experiment reports:

- Validation accuracy: accuracy on the validation split, used for model comparison before final test interpretation.
- Test accuracy: accuracy on the held-out test split.
- Test error rate: `1 - test accuracy`, shown to make small differences near high accuracy easier to compare.
- Training time: elapsed fitting time in seconds.

Hard Voting and Soft Voting reuse already fitted individual models. They do not retrain the base learners, so their additional `training_time_seconds` is recorded as `0.0` in this experiment. Their prediction times still include the time required to collect and combine predictions or probabilities from the fitted base models.

## 5. Main Results

The results table below is based on `results/tables/stage2_mnist_voting_results.csv`.

| Model | Validation Accuracy | Test Accuracy | Test Error Rate | Macro F1 Test | Training Time (s) |
|---|---:|---:|---:|---:|---:|
| Extra-Trees | 0.9608 | 0.9613 | 0.0387 | 0.9610 | 3.6878 |
| Random Forest | 0.9570 | 0.9572 | 0.0428 | 0.9569 | 3.6121 |
| Hard Voting | 0.9454 | 0.9493 | 0.0507 | 0.9488 | 0.0000 |
| Soft Voting | 0.9338 | 0.9347 | 0.0653 | 0.9340 | 0.0000 |
| Linear SVM-like SGD | 0.9062 | 0.9104 | 0.0896 | 0.9097 | 109.2042 |
| Logistic Regression | 0.8882 | 0.8923 | 0.1077 | 0.8910 | 19.6975 |

Extra-Trees performs best in this run, with the highest validation accuracy and test accuracy. Random Forest is close behind and is the second strongest model. Hard Voting and Soft Voting do not beat the strongest individual classifier. The linear models are weaker on raw MNIST pixel features than the tree ensembles.

## 6. Interpretation

### 6.1 Why Extra-Trees and Random Forest outperform linear models

Extra-Trees and Random Forest can capture nonlinear feature interactions among pixels. In handwritten digit recognition, the meaning of one pixel often depends on nearby or related pixels. Tree ensembles can split on combinations of pixel intensities and build many local decision rules.

The Linear SVM-like SGD classifier and Logistic Regression use linear decision boundaries in the raw pixel feature space. These models can still learn useful patterns, but they are less flexible for representing nonlinear digit shapes without additional feature engineering.

### 6.2 Why naive voting did not beat the best individual classifier

Voting helps only when base learners are both individually relevant and usefully diverse. In this experiment, Extra-Trees and Random Forest are much stronger than the linear models. Adding weaker models to a voting ensemble can dilute the predictions of the strongest classifier.

Hard Voting and Soft Voting both improve over the weaker linear models, but neither improves over Extra-Trees or Random Forest. This suggests that simple equal-weight voting is not the best way to combine these particular models.

### 6.3 Relevance and useful diversity

A good ensemble needs more than many models. It needs base learners that are accurate enough to contribute useful information and diverse enough to make different errors. Diversity that comes mainly from weaker predictions is not necessarily helpful.

These results connect directly to the ensemble learning idea that useful diversity matters. The next step is to learn how to combine model outputs instead of assigning every model equal influence.

## 7. Lessons Learned

- Ensemble learning is not automatically better than the best individual model.
- Simple voting is a useful baseline, but it is not a guaranteed improvement.
- Diversity must be useful, not merely different.
- Stage 3 should try stacking or blending to learn combination weights from validation predictions.

## 8. Resume Description

- Built a hands-on ensemble learning lab in Python using scikit-learn, including baseline classifiers, voting ensembles, validation/test evaluation, and reproducible result artifacts.
- Compared Random Forest, Extra-Trees, linear classifiers, hard voting, and soft voting on MNIST using accuracy, macro F1, test error rate, and runtime metrics.
- Analyzed why naive voting did not outperform the strongest tree ensemble and proposed stacking/blending as the next project stage.
