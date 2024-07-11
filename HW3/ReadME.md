this folder is my first theorical assignment. it is about concepts of deep learning. here is the list of questions and a brief explanation. about each. the latex source is in the folder called LatexMaterial. feel free to make changes in them.
- [x] Q1 Design a Neural Network: Design a neural network to generate Hamming codes for 4-bit inputs.
- [x] Q2 (∗) Design Simple Neural Network: Design a neural network with one hidden layer to implement a given function.
- [x] Q3 (∗) Vector Derivative: Determine the vector derivative of a composite function at a specific point.
- [x] Q4 (∗) Backpropagation Algorithm1: Calculate node values and perform one step of backpropagation for a two-layer neural network.
  - [x] Q4.1: Calculate the value at nodes \( ŷ, h1, h2 \) for given inputs.
  - [x] Q4.2: Execute one step of backpropagation for the given inputs and output.
  - [x] Q4.3: Calculate the updated weights for the hidden and output layers.

- [x] Q5 Back propagation in CNN - Convolution in each direction!: Perform backpropagation for a convolutional layer and derive the required derivatives.
- [x] Q6 Back propagation in CNN - an example: Perform backpropagation on a sequence of layers and write update rules for weights.
- [x] Q7 CNNs are universal approximators: Discuss if an equivalent MLP can be made from a CNN and under what conditions.
- [x] Q8 Backpropagation Algorithm2: Obtain the derivative of the loss function with respect to the first layer's weights using backpropagation.
- [x] Q9 (∗) Model Parameters: Determine the size of the kernel, number of trainable parameters, and multiplication operations for a two-layer convolutional network.
- [x] Q10 (∗) Receptive field: Determine a formula for the receptive field of a convolution layer in terms of kernel size and stride.

- [x] Q11 Optimizating Deep Learning Models: Discuss the momentum in optimization algorithms.
  - [x] Q11.1.1 Questions 1: Explain the Nesterov Accelerated Gradient (NAG) algorithm and its improvements over the Momentum algorithm.
  - [x] Q11.1.2 Questions 2: Explain the Adagrad algorithm and its improvements over the Momentum algorithm.

- [x] Q11.2 (∗) Approximate Newton Methods: Write the update rule of Newton’s Method and explain its working.
- [x] Q11.3 Adam Optimizer: Explain the Adam optimizer, its update rule, and improvements over the Momentum algorithm.
  - [x] Q11.3.1: Explain how Adam optimizer works, line by line.
  - [x] Q11.3.2: Explain why \(m_t\) has a bias towards zero in the early stages and how \( \hat{m}_t \) corrects this bias.

- [x] Q12 Quadratic Error Function: Analyze the quadratic error function and show the convergence of weights using gradient descent.

- [x] Q13 LSTM: Prove that the LSTM cell can learn to remember information over long sequences by showing the non-vanishing derivative of the cell state.

- [x] Q14 (∗) RNN: 
  - [x] Q14.1.1 Complex Dynamics of RNNs: Derive the Jacobian matrix of the hidden state transitions and discuss its role in stability and gradient issues.
  - [x] Q14.2.2 BPTT in Depth: Provide a detailed derivation of the Backpropagation Through Time (BPTT) process for RNNs.

- [x] Q15 GANs: Discuss the training process of GANs and related challenges.
  - [x] Q15.1.1 GANs, a game theoretic approach: Define GANs as a game, identifying players, actions, and payoffs.
  - [x] Q15.1.2 Questions 2: Explain the optimization problem for finding the Nash Equilibrium in GANs and its relation to training.
  - [x] Q15.1.3 Questions 3: Solve the GANs optimization problem using the calculus of variations to find a closed-form solution for \( \theta_D \).

  - [x] Q15.2 GANs, you don't want to train them!: Discuss the instability and challenges of training GANs.
    - [x] Q15.2.1 Questions 1: Explain why the first term of the GANs loss is not affected when updating generator parameters.
    - [x] Q15.2.2 Questions 2: Explain the mode collapse problem and its impact on GAN training.
    - [x] Q15.2.3 Questions 3: Explain the Wasserstein distance and how it addresses the issues in the GAN loss function.
