# Notes on Machine Learning

## Cross validation
Technique to evaluate a model by partitioning the original data into a training set and a validation set. A commonly used kind of cross validation is the <strong>k-fold cross validation</strong>. In this specific setting, original data is divides into k-folds, training model on the k-1folds and testing on the kth fold. This results in k different models, which is averaged to get an overall model performance.

Source: 
- https://github.com/ShuaiW/data-science-question-answer

<a href="#top">Back to top</a>

## Feature Importance in linear models and tree-based models
In linear models, feature importance can be obtained by calculating the scale of the coefficients.

In tree-based models, especially in the <strong>Random Forest</strong> setting, feature importance can be obtained by averaging the depth at which it appears across all trees in the forest. Important features are more likely to appear closer to the root of the tree in tree-based models.

Source:
- https://github.com/ShuaiW/data-science-question-answer

<a href="#top">Back to top</a>
 
## Mean Squared Error (MSE) vs. Mean Absolute Error (MAE)
Similarity: Both measure average prediction of a model.

MSE is the average of the sums of residual squares and MAE is the average of the sums of absolute value of the residuals. MSE gives higher weights to larger errors while MAE gives equal weights to all the errors. MSE is continuously differentiable, but MAE is not.

Source:
- https://github.com/ShuaiW/data-science-question-answer

 <a href="#top">Back to top</a>
 
## Bias vs. Variance
<strong>Bias</strong> is the inability of the model to capture the true relationship of the data. Bias is defined as the difference between the estimated value of the parameter and the true parameter. If the difference is 0, then we have an unbiased model (i.e. the model is able to interpret the data very well). On the other hand, if the bias is big, then the model does not interpret the data most of the time.

In terms of comparing bias with training and test sets, if the model has a low bias, then the model fits the training set very well (i.e. low sums of squares of residuals). However, conversely, the model will not fit the test set well (i.e. this is because the model is close to being unbiased towards the training set).

<strong>Variance</strong> is the difference in fits between different datasets (i.e. test set vs. training set). It is important to note that if the model's bias is low (model fits the test set well), there is a trade-off, so model's variance will be high (model's sums of squares are high towards test set). Conversely, if model's bias is high, then the model's variance will be low, meaning that the model fits the test set well compared to the training set.

Hence, we do not want our model to have high variance, neither a high bias. We just want the right amount of bias and variance so that our model will optimally fit both the test set and the training set on the sweet-spot (an ideal model would have both low variance and bias). Three commonly used methods to find the sweet-spot are: <strong>regularization</strong>, <strong>boosting</strong> and <strong>bagging</strong>. Examples include <strong>L1/L2 Regularization</strong>,<strong>AdaBoost/XGBoost</strong>,<strong>Random Forest</strong> accordingly.

When the model has low bias, we say that the model is an <strong>overfit</strong>. Likewise, when the model has high bias, we say that the model is an <strong>underfit</strong>.

A <strong>simple model</strong> is a model that has <strong>low bias</strong> and <strong>high variance</strong>. On the other hand, a <strong>complex model</strong> is a model that has <strong>low variance</strong> and <strong>high bias</strong>. For example, OLS (Ordinary Least Squares) model will be a complex model with high variance and low bias. By applying regularization techniques, we are able to adjust our model to have lower variance and higher bias to get close to the sweet spot.

Source:
- https://github.com/ShuaiW/data-science-question-answer
- https://www.youtube.com/watch?v=EuBBz3bI-aA&t=211s

<a href="#top">Back to top</a>

## Would adding more data address underfitting?
Recall that <strong>underfitting</strong> is when model has high bias and low variance in other words, the model is too simple to interpret the data. This is a problem of the model rather than the data itself. We can address this issue by increasing model complexity:
  1. Increase depth for tree based methods.
  2. Add higher order coefficients for linear models.
  3. Add more layers for neural networks.

<a href="#top">Back to top</a>

## L1 vs L2 Regularization
<strong>L1 (Lasso)</strong> and <strong>L2 (Ridge)</strong> Regressions are methods to prevent model overfitting by imposing penalty terms. L1 shrinks some coefficients to zeros, resulting in variable selection while L2 can only shrink coefficients close to zero. 

<strong>L1 (Lasso) Regression</strong> fits a new line on the dataset that does not fit the training set too well (introduce a small amount of bias and thereby also achieving a significant decreasing the variance). In other words, with a slightly worse fit to the training set, we will obtain a model that is better fit to the test set.
  - Given an OLS equation, we add a penalty term (i.e. lambda x absolute value of the slopes) to punish the bias.
  - Convention is that we use 10-fold CV (Cross Validation) to determine optimal value of lambda that outputs lowest variance.
  - L1 can shrink some slopes to zero leading to variable selection.

<strong>L2 (Ridge) Regression</strong> fits a new line on the dataset that does not fit the training set too well as well. 
- Given an OLS equation, we add a penalty term (i.e. lambda x slope^2) to punish the bias.
- Convention is that we use 10-fold CV (Cross Validation) to determine optimal value of lambda that outputs lowest variance.
- When we have more features than the number of samples, L2 can improve model predictions by making predictions less sensitive to training set by adding penalties.

If all the features are correlated with a label, L2 outperforms L1. On the other hand, if only a subset of features are correlated with a label, L1 outperforms L2, as some coefficients are shrunk to 0.

Source:
- https://github.com/ShuaiW/data-science-question-answer
- https://www.youtube.com/watch?v=Q81RR3yKn30&t=713s
- https://www.youtube.com/watch?v=NGf0voTMlcs

<a href="#top">Back to top</a>

## Elastic Net
<strong>Elastic Net</strong> is a regularizer combined by L1 and L2. It overcomes the limitations of L1, i.e., given large p and small n, L1 selects at most n features before it saturates. If there are highly correlated group of variables, L1 tends to select one out of that group and ignores others.
  - There are two lambdas (i.e. lambda_1 and lambda_2) that are multiplied with absolute value of slopes and slope^2 accordingly.
- Because there are two lambdas that control the shrinkage, this leads to increased bias and poor predictions. Hence to improve performance, we multiply estimated coefficients by (1+lambda).

Source:
- https://www.youtube.com/watch?v=1dKRdX9bfIo

<a href="#top">Back to top</a>

## Covariance and Correlation
Both determine the relationship and measure dependency between two random variables. <strong>Covariance</strong> indicates the direction between the relationship between two variables (i.e. positive/negative/neutral). Covariance is not a scaled measure while <strong>Correlation</strong> is a standardized measure that is not sensitive to scale of the data.

<a href="#top">Back to top</a>

## Activation function
<strong>Activation function</strong> generates outputs after processing inputs in a node or a neuron.

Linear: Regression

Non-linearity: ReLU (Rectified Linear Unit) is often used. Also use Leaky ReLU to address dead ReLU issue.
  - Nonlinearity is a relationship which cannot be explained as a linear combination of its variable inputs. In other words, the outcome does not change in proportion to a change in any of the inputs.
  - Leaky ReLU allows small, positive gradient when the unit is NOT active (f(x) = x, x > 0 and f(x) = 0.01x, else).

Multi-Class: Softmax (generalized version of the logistic function that 'squishes' a k-dimensional vector z of arbitrary real values to a k-dimensional vector of real numbers in [0,1] that adds up to 1).

Binary: Sigmoid, Step function

Source: 
- https://github.com/ShuaiW/data-science-question-answer

<a href="#top">Back to top</a>

## Decision Tree
Consists of <strong>root node</strong>, <strong>internal node</strong> and <strong>leaf node</strong>.

Feature is <strong>Impure</strong> when none of the leaf nodes are able to predict 100% of a given outcome (i.e. 100% Yes, heart disease). <strong>Impurity</strong> is a way to measure which separation is the best. <strong>Gini</strong> is the most commonly used one.

<strong>Categorical features</strong>

1: Calculate the impurity scores of all values. Root node is the feature with the lowest impurity value

2: If the node itself has the lowest impurity score, then there is no point in branching out so it becomes a leaf node.

3: If separating data results in an improvement, then pick the feature with the lowest impurity value.

Feature with the lowest gini impurity score will be assigned as the root node. On the next branch, again, feature with the lowest gini impurity score will be assigned to the internal node.

<strong>Numerical features</strong>

1. Order the feature/column in increasing order and calculate the average value for all adjacent rows/entries.

2. Calculate the impurity values for each average values.

Source:
- https://www.youtube.com/watch?v=7VeUPuFGJHk&t=2s

<a href="#top">Back to top</a>

## Random Forest
<strong>Bootstrapping</strong> is any test metric that relies on random sampling with replacement.
1. Take bootstrapped sample for the data (repetition is possible).
2. Take only a subset of the bootstrapped sample columns each step to create decision trees.
3. Repeat steps 1 & 2 'n' times.
4. Run test samples on the newly built n trees and pick the answer with the most votes.

Usually 1/3 of the dataset will be left out and will be used as the 'out-of-bag' dataset that will be used for testing.

<strong>Out-of-bag error</strong> is the proportion of out-of-bag samples that were incorrectly classified. Typically use sqrt(number of features) to tune for the number of sample features to be picked each step and try few settings above and below.

<strong>Bagging</strong>: bootstrapping data + using the aggregate to make decision. Used to reduce variance'

We can use out-of-bag error to adjust the size of the subset we take from the bootstrapped sample.

Source:
- https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
- https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
- https://www.youtube.com/watch?v=nyxTdL_4Q-Q

<a href="#top">Back to top</a>

## AdaBoost
In contrast to Random Forest, AdaBoost has a set amount of depth, that is, just a node with 2 leaves. Whereas in RF, each trees are independent of the other, an error from the first tree will affect the following trees. Moreover, in RF the trees' final classification have same equal weights, but in AdaBoost, stumps have varying weights (stump with higher error will have higher weights).

<strong>Stumps</strong> are trees with 1 node and 2 leaves. They are <strong>weak learners</strong> because it only uses one feature.

3 ideas behind AdaBoost:
1. AdaBoost combines a lot of weak learners to classify.
2. Some stumps weigh more than others.
3. Each stump is made by taking previous stump's error into account.

For the first stump, each sample's weights are equal. Ada Boost works by putting more weights on stumps that higher errors and less weights on those that had less errors.

<strong>Total Error of the stump</strong> is the sum of the weights associated with incorrectly classified samples.

Source:
- https://www.youtube.com/watch?v=LsK-xG1cLYA&t=749s

<a href="#top">Back to top</a>

## Bagging
We use bagging to address overfitting, which reduces variance of the meta learning algorithm. 

Bagging is when sampling is performed with replacement. If sampling is performed without replacement, it is called <strong>pasting</strong>. Bagging is famous for its boost in performance and also because individual learners can be trained in parallel and scale well.

<a href="#top">Back to top</a>

## Voting
<strong>Hard Voting</strong> is where a model is selected from an ensemble to make the final prediction by simple majority vote for accuracy.

<strong>Soft Voting</strong> can only be done when all the classifiers can calculate probabilities of the outcomes. Soft Voting is at its best when we average all the probabilities.

Source:
- https://towardsdatascience.com/ensemble-learning-in-machine-learning-getting-started-4ed85eb38e00

<a href="#top">Back to top</a>

## Stacking
The idea of stacking is to learn several different weak learners and combine them by training meta-model to output predictions based on multiple predictions returned by these weak models.

Steps: 
1. Split data into two folds.
2. Choose weak learners and fit them to the data from the first fold.
3. For each of the weak learners, make prediction for observations in the second fold.
4. Fit meta-model on the second fold, using predictions made by the weak learners as inputs.

Source:
- https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205

<a href="#top">Back to top</a>

## Ensemble Methods
<strong>Meta Algorithm</strong> are algorithms that wrap and executes other algorithms and might feed them input data or use their output data. A common goal is to achieve better task performance.

<strong>Ensemble Methods</strong> are meta-algorithms that combine several ML techniques into 1 predictive model in order to decrease variance (<strong>bagging</strong>), bias (<strong>boosting</strong>), or improve predictions (<strong>stacking</strong>).

<strong>Sequential Ensemble methods:</strong> Base learners are generated sequentially (i.e. AdaBoost)
  - Motivation: Exploit the dependence between the base learners.
  - Overall performance can be boosted by weighing previously mislabled examples with higher weight.

<strong>Parallel Ensemble methods:</strong> ensemble methods where base learners are generated in parallel (i.e. Random Forest)
  - Motivation: exploit independence between base learners since error can be reduced dramatically by averaging.

Most ensemble methods use a single base learning algorithm to produce homogeneous base learners, i.e. learners of the same type, leading to homogeneous ensembles.

There also exists heterogeneous ensembles/learners: base learners have to be as accurate as possible and diverse as possible.

<strong>Further resources on implementing Ensemble Learning:</strong> https://towardsdatascience.com/ensemble-learning-in-machine-learning-getting-started-4ed85eb38e00
<strong>Further resources on Ensemble Learning</strong>:https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205

Source:
- https://github.com/ShuaiW/data-science-question-answer/

<a href="#top">Back to top</a>

