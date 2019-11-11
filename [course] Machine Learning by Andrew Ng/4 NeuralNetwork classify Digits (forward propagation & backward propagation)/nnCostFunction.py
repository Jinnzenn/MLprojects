"""
NNCOSTFUNCTION Implements the neural network cost function for a two layer neural network which performs classification
[J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels,X, y, lambda ) computes the cost and gradient of the neural network.
The parameters for the neural network are "unrolled" into the vector nn_params and need to be converted back into the weight matrices.
The returned parameter grad should be a "unrolled" vector of the partial derivatives of the neural network.
Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
"""

import numpy as np


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lambda): #输入特征数量
    Theta1 = reshape(nn_params[1:hidden_layer_size] * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1))
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):),num_labels, (hidden_layer_size + 1))

    # Setup some useful variables
    (m,n) = X.shape # 样本数量和

    # You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # 记得加上 bias unit 偏置单元
    X = np.c_(np.ones(m),X)

    # 将y值转成1 * 10的矩阵
    Y = np.zeros(m, num_labels)
    for i in range(m)
        Y(i, y(i)) = 1

    # Forwardpropagation

    a2 = sigmoid(np.dot(Theta1,X)
    a2 = np.c_(np.ones(a2.shape[1]),a2)
    a3 = sigmoid(np.dot(Theta2.X)

    term1 = -np.dot(y,np.log(a3))
    term2 = -np.dot(1-y,np.log(1-a3))
    term3 = (np.dot(Theta1[1:],Theta1[1:]) + np.dot(Theta2[1:],Theta2[1:])) * lambda
    J = (1/m) * (term1 + term2) + (1/2m) * term3  # regularized cost_function

    # Backpropagation
    d1 = zeros(size(Theta1))
    d2 = zeros(size(Theta2))

    theta1_wtbias = Theta1
    theta1_wtbias(:, 1) = 0

    theta2_wtbias = Theta2
    theta2_wtbias(:, 1) = 0

    for t in range(m)
        yt = Y(t,:)
        a3t = a3(t,:)
        a2t = a2(t,:)
        a1t = X(t,:)
        delta3 = a3t - yt;
        delta2 = delta3 * Theta2. * (a2t. * (1 - a2t));
        delta2 = delta2(2:end);
        d2 = d2 + delta3‘ *a2t;
        d1 = d1 + delta2‘ *a1t;

    # regularized theta
    Theta1_grad = Theta1_grad + d1 / m + lambda *theta1_wtbias / m
    Theta2_grad = Theta2_grad + d2 / m + lambda *theta2_wtbias / m

    # Unroll gradients
    grad = [Theta1_grad(:); Theta2_grad(:)]