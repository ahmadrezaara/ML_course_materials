this folder is my first theorical assignment. it is about concepts of machine learning. here is the list of questions and a brief explanation. about each. the latex source is in the folder called LatexMaterial. feel free to make changes in them.
- [x] Q1 Easy peasy lemon squeezy: Calculate probabilities using logistic regression for predicting student scores based on hours of study and GPA.
  - [x] Q1.1: Calculate the probability that a student with 80 hours of study and a GPA of 18 can get a score of 20.
  - [x] Q1.2: Determine the required study hours for a student with a GPA of 16 to achieve a score of 20 with a 90% probability.

- [x] Q2 Multi-class Logistic Regression: Extend logistic regression to multi-class settings and calculate probabilities and decision boundaries.
  - [x] Q2.1: Determine the model implications for $( P(Y = y_K | X) )$.
  - [x] Q2.2: State the classification rule for multi-class logistic regression.
  - [x] Q2.3: Draw a training data set with three labels and the resulting decision boundary.
  - [x] Q2.4: Find the log-likelihood for observed data samples.
  - [x] Q2.5: Calculate the gradient of the log-likelihood function with L2 regularization.
  - [x] Q2.6: Write and explain the update rule for the data set.

- [x] Q3 Overfitting and Regularized Logistic Regression: Analyze overfitting and derive gradient ascent rules for logistic regression with regularization.
  - [x] Q3.1: Plot the sigmoid function for increasing weights and explain overfitting.
  - [x] Q3.2: Derive gradient ascent update rules for logistic regression with a Gaussian prior.

- [x] Q4 *Lasso Regression: Explore the Lasso regression method and its impact on sparsity.
  - [x] Q4.1: Show how whitening the dataset causes feature independence and express $( J_\lambda(w) )$.
  - [x] Q4.2: Find $( w_i )$ if $( w_i \geq 0 )$.
  - [x] Q4.3: Find $( w_i )$ if $( w_i < 0 )$.
  - [x] Q4.4: Determine the conditions under which \( w_i = 0 \) and compare to Ridge regression.

- [x] Q5 Naive Bayes: Classify data using Naive Bayes with features from Bernoulli and Gaussian distributions.
  - [x] Q5.1: Find $( p(Y|X_1, X_2) )$ for given values.
  - [x] Q5.2: Find $( p(Y|X_1) )$ for a given value.
  - [x] Q5.3: Find $( p(Y|X_2) )$ for a given value.
  - [x] Q5.4: Justify the observed pattern in probabilities.

- [x] Q6 Multivariate Least Squares: Extend least squares to handle multiple outputs.
  - [x] Q6.1: Write the cost function $( J(\Theta) )$ in matrix-vector notation.
  - [x] Q6.2: Find the closed-form solution for $( \Theta^* )$ that minimizes $( J(\Theta) )$.
  - [x] Q6.3: Compare the multivariate solution to independent least squares problems.

- [x] Q7 LDA Vs. LR: Compare Linear Discriminant Analysis (LDA) and Linear Regression (LR) in classification.
  - [x] Q7.1: Show the LDA classification condition for a sample.
  - [x] Q7.2: Relate the least squares estimate $( \beta_1 )$ to the LDA coefficient.
  - [x] Q7.3: Conclude the equivalence of LDA and LR.

- [x] Q8 QDA: Find the decision boundary using Quadratic Discriminant Analysis (QDA) for circularly distributed samples.
  - [x] Q8: Calculate the decision boundary for given sample distributions.

- [x] Q9 Hana: Analyze the correlation between variables and residuals in regression.
  - [x] Q9.1: Show the correlation of variables with residuals as a function of $( \alpha )$.
  - [x] Q9.2: Demonstrate the monotonic decrease of correlations towards zero.

- [x] Q10 Ridge Regression: Explore ridge regression and its relationship with Gaussian priors.
  - [x] Q10.1: Show that ridge regression estimate is the mean of the posterior distribution under a Gaussian prior.
  - [x] Q10.2: Obtain ridge regression estimates using ordinary least squares on augmented data.

- [x] Q11 Estimate: Compute estimators for the mean of a normal distribution.
  - [x] Q11.1: Compute the MLE for the mean $( \mu )$.
  - [x] Q11.2: Compute the MAP for the mean $( \mu )$ assuming a normal prior.
  - [x] Q11.3: Analyze the behavior of estimators as the sample size increases towards infinity.
