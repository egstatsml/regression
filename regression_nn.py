#!/usr/bin/env python
'''
A regression example using TensorFlow library.

Author: Ethan Goan

Modified from original source by Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

MIT Licence
'''


import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def forward_pass(X, W_1, b_1, W_2, b_2, W_o, b_o):
# Construct a linear model
    A_1 = tf.nn.tanh(tf.multiply(W_1, X) + b_1)
    A_2 = tf.nn.tanh(tf.matmul(W_2, A_1) + b_2)
    return tf.matmul(W_o, A_2) + b_o


def initialise_parameters(hidden_layer_dims):
    #first hidden layer
    W_1 = tf.Variable(rng.randn(hidden_layer_dims[0], 1),
                      name="weight_1", dtype=tf.float64)
    b_1 = tf.Variable(rng.randn(hidden_layer_dims[0], 1),
                      name="bias_1", dtype=tf.float64)
    #second hidden layer
    W_2 = tf.Variable(rng.randn(hidden_layer_dims[1], hidden_layer_dims[0]),
                      name="weight_2", dtype=tf.float64)
    b_2 = tf.Variable(rng.randn(hidden_layer_dims[1], 1),
                      name="bias_2", dtype=tf.float64)
    #output layer
    W_o = tf.Variable(rng.randn(1, hidden_layer_dims[1]),
                      name="weight_o", dtype=tf.float64)
    b_o = tf.Variable(rng.randn(1, 1),
                      name="bias_o", dtype=tf.float64)

    return W_1, b_1, W_2, b_2, W_o, b_o


if __name__ == "__main__":
    rng = np.random
    # hyperparameters
    learning_rate = 0.01
    training_epochs = 5000
    display_step = 50
    hidden_layer_dims = [20, 20]
    
    # Training Data
    train_X = np.load('x.npy').ravel()
    train_Y = np.load('y.npy').ravel()
    # test data
    test_X = np.load('x_s.npy').ravel()
    test_Y= np.load('y_test.npy').ravel()
    n_samples = train_X.shape[0]

    # tf Graph Input
    X = tf.placeholder("float64")
    Y = tf.placeholder("float64")

    #Initialise our parameters
    W_1, b_1, W_2, b_2, W_o, b_o = initialise_parameters(hidden_layer_dims)
    
    # Construct a linear model
    pred = forward_pass(X, W_1, b_1, W_2, b_2, W_o, b_o)

    # Defining our cost function
    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are
    #  trainable=True by default
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Training cost=", training_cost, '\n')

        # Done training, lets do some testing
        print("Testing... (Mean square loss Comparison)")
        testing_cost = sess.run(
            tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
            feed_dict={X: test_X, Y: test_Y})  # same function as cost above
        print("Testing cost=", testing_cost)
        print("Absolute mean square loss difference:", abs(
            training_cost - testing_cost))

        plt.plot(test_X, test_Y, 'b', label='Testing data')
        y_nn = sess.run(forward_pass(test_X, W_1, b_1, W_2, b_2, W_o, b_o))
        plt.plot(test_X,
                 y_nn.ravel(), #the ravel just makes it one dimensional
                 'r', label='Fitted line')
        plt.legend()
        plt.savefig('regression_nn.png')
        #save the regression data
        np.save('x_nn', test_X)
        np.save('y_nn', y_nn)
