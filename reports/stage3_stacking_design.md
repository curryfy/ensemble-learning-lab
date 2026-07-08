# Stage 3: MNIST Stacking / Blending Design

## 1. Motivation

Stage 2 showed that naive hard voting and soft voting did not outperform the strongest individual classifier on the MNIST experiment. Extra-Trees was the best model in that run, while equal-weight voting mixed strong and weaker learners without learning which models should receive more influence.

Stage 3 will test whether a learned blender can combine base model outputs more effectively than equal-weight voting. The goal is to move from simple voting rules to a small supervised meta-model that learns how much to trust each base learner.

## 2. Connection to ESL Ensemble Learning

This stage connects to several ensemble learning ideas from ESL Chapter 16:

- Model averaging: combining predictions from multiple fitted learners.
- Combining learners: using different model families to capture different patterns in the data.
- Relevance-diversity tradeoff: useful ensembles need base learners that are accurate enough to help and diverse enough to make complementary errors.
- Learned combination weights: replacing equal influence with weights learned from validation predictions.

Hard voting and soft voting are simple model averaging baselines. Stacking or blending extends this idea by training a second-level model to combine the base learners. Instead of assuming every model should contribute equally, the blender can learn to emphasize stronger or more reliable models.

## 3. Planned Base Models

Stage 3 will use the same base model family from Stage 2:

- Extra-Trees
- Random Forest
- Linear SVM-like SGD classifier
- Logistic Regression

Extra-Trees and Random Forest provide strong nonlinear tree-ensemble baselines. Logistic Regression provides a simpler linear probabilistic baseline. The Linear SVM-like SGD classifier provides a fast margin-based linear model.

Only models with probability outputs can directly contribute probability features. Extra-Trees, Random Forest, and Logistic Regression can provide `predict_proba` outputs. If the hinge-loss SGD classifier does not provide `predict_proba`, it may need probability calibration or may contribute decision scores instead.

## 4. Data Splitting Strategy

The intended split is:

- Training set: fit the base models.
- Validation set: construct blender training data from base model predictions.
- Test set: evaluate the final individual models, voting ensembles, and stacking/blending ensemble.

Avoiding data leakage is essential. The validation labels can be used to train the blender, but the test set must not be used to train either the base models or the blender. The test set should be touched only once for final evaluation.

## 5. Blender Feature Construction

Each fitted base model produces class probabilities or scores on the validation set. These outputs become the input features for the blender.

For MNIST with 10 classes and `M` base models, the blender input has approximately `M x 10` features if probabilities are used. For a base model `m`, write its predicted class-probability vector as:

`p_m(x) = (p_{m,0}(x), ..., p_{m,9}(x))`

The full blender feature vector is the concatenation of the base model outputs:

`z(x) = [p_1(x), ..., p_M(x)]`

If one base model contributes decision scores instead of probabilities, those scores should be treated carefully because they may not be on the same scale as calibrated probabilities.

## 6. Planned Blender Model

The first blender should be a simple multinomial Logistic Regression model.

The blender should stay simple because the goal is to learn combination weights, not to overfit the validation set. A complex blender could memorize validation-specific patterns and give an overly optimistic impression of ensemble performance. Logistic Regression is a good first meta-model because it is interpretable, regularized, and well suited for multiclass classification.

## 7. Evaluation Plan

Stage 3 should compare:

- Best individual classifier
- Hard voting
- Soft voting
- Stacking/blending

The evaluation should use the same metrics as Stage 2:

- Validation accuracy
- Test accuracy
- Test error rate
- Macro F1
- Training time

The final report should make clear which time measurements belong to base model fitting, blender fitting, and prediction. The comparison should avoid claiming improvement unless the held-out test metrics support it.

## 8. Expected Learning Questions

- Can stacking beat naive voting?
- Does the blender learn to down-weight weaker models?
- Does stacking improve over Extra-Trees, or only over voting?
- How sensitive is the result to validation set size?
- Are probability features enough, or do decision-score features from the SVM-like model help?
- Does the blender improve macro F1 as well as accuracy?

## 9. Implementation Checklist

- [ ] Create a Stage 3 notebook without modifying the Stage 1 or Stage 2 notebooks.
- [ ] Reuse the MNIST split strategy from Stage 2.
- [ ] Fit the base models only on the training set.
- [ ] Generate validation predictions or probabilities from each fitted base model.
- [ ] Build the blender feature matrix from validation-set base model outputs.
- [ ] Train a multinomial Logistic Regression blender on the validation-set blender features.
- [ ] Generate test-set blender features using the fitted base models.
- [ ] Evaluate the blender on the test set only after training is complete.
- [ ] Compare the blender against the best individual classifier, hard voting, and soft voting.
- [ ] Save a results table and any figures as new Stage 3 artifacts.
- [ ] Document whether stacking improves over voting and whether it improves over Extra-Trees.
