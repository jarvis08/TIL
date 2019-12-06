# Loss Functions

## Definition of loss function

<br>

### Why negative?

<br>

### Loss functions

<br>

<br>

## In Regression.

### Mean Square Error / Quadratic Loss / L2 Loss

$MSE = \frac{\sum^{n}_{i=1}(y_{i} - \hat{y}_{i})^2}{n}$

<br>

### L2 Regularization

<br>

### Mean Absolute Error/L1 Loss

$MAE = \frac{\sum^{n}_{i=1}|y_{i} - \hat{y}_{i}|}{n}$

<br>

### L1 Regularization

<br>

### Mean Bias Error

$MSE = \frac{\sum^{n}_{i=1}(y_{i} - \hat{y}_{i})}{n}$

<br>

<br>

## In Classification.

### Hinge Loss/Multi class SVM Loss

$SVMLoss = \sum_{j≠y_{i}}max(0, s_{j} - s_{y_{i}}+1)$

<br>

### Cross Entropy Loss/Negative Log Likelihood

$CrossEntropyLoss = -(y_{i}log(\hat{y}_{i}) + (1-y_{i})log(1-\hat{y}_{i}))$

$CrossEntropyLoss = -\sum_{x}P(x)logQ(x)$

