this folder is my first theorical assignment. it is about concepts of kernel methods and SVM in machine learning. here is the list of questions and a brief explanation. about each. the latex source is in the folder called LatexMaterial. feel free to make changes in them.
- [x] Q1 (∗) Representer Theorem:
  - [x] Q1.1: Explain the meaning of Hilbert Space, Reproducing Kernel Hilbert Space, Reproducing Kernel, and Mercer’s Theorem.
  - [x] Q1.2: Write \( f \in HK \) using Mercer's theorem in a specific form.
  - [x] Q1.2.1: Write \( f \) as a dot product of \(\Phi_k\)’s using the reproducing kernel property.
  - [x] Q1.3: Find a lower bound on the regularization term \( R(\|f\|) \).
  - [x] Q1.4: Jointly optimize both the loss terms and the penalty function to prove the representer’s theorem.
  - [x] Q1.5: Compare the representer theorem solution to the final SVM solution.

- [x] Q2 (∗) Neural Networks Can be Seen as (almost) GPs!:
  - [x] Q2.1: Show that the expected output of an MLP with one hidden layer is zero.
  - [x] Q2.1.2: Show the covariance of the output for two different inputs.
  - [x] Q2.1.3: Argue that as \( H \to \infty \), the network output converges to a multivariate Gaussian distribution.
  - [x] Q2.2: Explain Neural Tangent Kernels in two paragraphs.

- [x] Q3 (∗) SVM:
  - [x] Q3.1: Discuss the position and classification of points relative to the margin based on slack values.
  - [x] Q3.2: Find transformations of features \( X1 \) and \( X2 \) for linearly separable datasets.

- [x] Q4 Dimensionality Reduction using PCA: Find the first principal component and project data points onto it.

- [x] Q5 Reconstruction Error:
  - [x] Q5.1: Prove \( \|x̂i - x̂j\|^2 = \|zi - zj\|^2 \).
  - [x] Q5.2: Prove \( \sum_{i=1}^{n} \|xi - x̂i\|^2 = (n-1) \sum_{i=k+1}^{p} \lambda_i \).

- [x] Q6 (∗) Clustering:
  - [x] Q6.1: Draw the boundary found by K-means for \( K=2 \) and discuss its meaningfulness.
  - [x] Q6.2: Illustrate the importance of careful initial point selection in K-means clustering.
  - [x] Q6.3: Explain K-means++ algorithm, WCSS, and elbow method.

- [x] Q7 Mixture Models:
  - [x] Q7.1: Explain how the MM algorithm handles non-convex optimization objectives.
  - [x] Q7.2: Find responsibilities in the E-step of Soft EM and class predictions in the E-step of Hard EM for a specific distribution.
  - [x] Q7.2.1: Modify the E-step of Soft EM to incorporate additional information when true labels are observed.

- [x] Q8 EM Algorithm for GMM and CMM:
  - [x] Q8.1: Compute estimates of parameters for Gaussian Mixture Models using EM algorithm.
  - [x] Q8.2: Compute estimates of parameters for Categorical Mixture Models using EM algorithm.

- [x] Q9 Advanced Hard-Margin SVM with Dual Problem and Kernel Methods:
  - [x] Q9.1: Define the primal optimization problem for a hard-margin SVM.
  - [x] Q9.2: Derive the dual optimization problem from the primal formulation.
  - [x] Q9.3: State the KKT conditions and explain the significance of support vectors.
  - [x] Q9.4: Use the kernel trick to replace the inner product in the dual problem and derive the dual optimization problem with a Gaussian RBF kernel.
  - [x] Q9.5: Solve the dual problem and interpret the results, explaining the geometric interpretation of the decision boundary.

- [x] Q10 Advanced Soft-Margin SVM with Regularization and Kernel Methods:
  - [x] Q10.1: Define the primal optimization problem for a soft-margin SVM with regularization and slack variables.
  - [x] Q10.2: Derive the dual problem for the soft-margin SVM.
  - [x] Q10.3: Discuss the application of a soft-margin SVM to a real-world classification problem and the process of parameter selection.
  - [x] Q10.4: Formulate the dual problem for a kernelized soft-margin SVM using the polynomial kernel.
  - [x] Q10.5: Explain how to evaluate the performance of a soft-margin SVM model and interpret the decision boundary.
