'''
This is a multilayer perceptron with a single hidden layer used to learn surface function from given 3D set of points.
Here I use dropout to avoid overfitting, simple mean squared cost (the fastest convergence in my case).
Plots the evaluation surface using matplotlib. For testing I sample from Gaussian multivariate distribution and try to
learn it with single-hidden layer neural net
'''

import random

import numpy
import scipy.stats
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats
# from mpl_toolkits.mplot3d import Axes3D

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def slice_data(input, expected):
    '''
    Splits data into three subsets
    '''
    training_index = int(len(input) * perc_train)
    validation_index = int(len(input) * perc_test) + training_index

    pm_x_fit = input[:training_index, :]
    pm_y_fit = expected[:training_index]

    pm_x_test = input[training_index:validation_index, :]
    pm_y_test = expected[training_index:validation_index]

    pm_x_evaluate = input[validation_index:, :]
    pm_y_evaluate = expected[validation_index:]

    return pm_x_fit, pm_y_fit, pm_x_test, pm_y_test, pm_x_evaluate, pm_y_evaluate


def gaussian_multivariate_distribution(X, Y):
    Z = []
    dist = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1, 0],
                                                             [0, 1]])
    for x, y in zip(X, Y):
        Z.append(dist.pdf([x, y]))
    return Z


def get_multilayer_perceptron(x, dropout_keep_prob_fn):
    # Store layers weight & bias

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden], dtype=tf.float64)),
        'h2': tf.Variable(tf.random_normal([n_hidden, n_hidden_2], dtype=tf.float64)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], dtype=tf.float64))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden], dtype=tf.float64)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], dtype=tf.float64)),
        'out': tf.Variable(tf.random_normal([n_classes], dtype=tf.float64))
    }

    # Hidden layer with relu activation
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1'])), dropout_keep_prob_fn)
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), dropout_keep_prob_fn)
    # layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer


# Parameters
learning_rate = 0.01
training_epochs = 300
display_step = 10
n_input = 2
n_classes = 1
n_hidden = 12
n_hidden_2 = 12
perc_train = 0.5
perc_test = 0.5
perc_eval = 0.0
min_step_size_train = 10 ** -8

x = numpy.asarray([random.uniform(-2, 2) for _ in range(5000)])
y = numpy.asarray([random.uniform(-2, 2) for _ in range(5000)])
inp = numpy.column_stack((x, y))
outp = numpy.asarray(gaussian_multivariate_distribution(x, y))  # evaluation of the function on the grid

n_samples = outp.shape[0]

# tf Graph Input
X = tf.placeholder("float64", [None, n_input])
Y = tf.placeholder("float64", [None, n_classes])

dropout_keep_prob = tf.placeholder(tf.float64)

x_fit, y_fit, x_test, y_test, x_evaluate, y_evaluate = slice_data(inp, outp)

# Construct a linear model
pred = get_multilayer_perceptron(X, dropout_keep_prob)

# Cost functions

cost = (tf.reduce_sum(tf.pow(pred - Y, 2)) / n_samples)
# cost = tf.reduce_mean(-tf.reduce_sum(*tf.log(pred), reduction_indices=1))
# cost = tf.nn.l2_loss(tf.subtract(pred, Y))+ beta * tf.nn.l2_loss(weights['h1']) + \
#        beta * tf.nn.l2_loss(weights['out'])
# cost = tf.nn.l2_loss(tf.subtract(pred, Y))


# Optimisers
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
evaluate_predictions_x = numpy.asarray([random.uniform(-2, 2) for _ in range(1000)])
evaluate_predictions_y = numpy.asarray([random.uniform(-2, 2) for _ in range(1000)])


evaluate_predictions_input = numpy.column_stack((evaluate_predictions_x, evaluate_predictions_y))

# Start training
with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=8, intra_op_parallelism_threads=8)) as sess:
    # Run the initializer
    sess.run(init)

    # Fit all training data
    last_cost = min_step_size_train + 1

    for epoch in range(training_epochs):
        # Trainning data
        for i in range(len(x_fit)):
            batch_x = numpy.reshape(x_fit[i, :], (1, n_input))
            batch_y = numpy.reshape(y_fit[i], (1, n_classes))

            # We want to use dropout only in the training stage
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 0.9})

        # Calculating data error
        c = 0.0
        for i in range(len(x_test)):
            batch_x = numpy.reshape(x_test[i, :], (1, n_input))
            batch_y = numpy.reshape(y_test[i], (1, n_classes))
            # Run Cost function
            c += sess.run(cost, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 1.0})

        c /= len(x_test)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.30f}".format(c))

        if abs(c - last_cost) < min_step_size_train:
            break
        last_cost = c

    nn_predictions = numpy.array([])

    for i in range(len(evaluate_predictions_input)):
        batch_x = numpy.reshape(evaluate_predictions_input[i, :], (1, n_input))
        nn_predictions = numpy.append(nn_predictions, sess.run(pred, feed_dict={X: batch_x, dropout_keep_prob: 1.0})[0])
    print("Optimization Finished!")

nn_predictions.flatten()

X, Y = numpy.meshgrid(evaluate_predictions_x, evaluate_predictions_y)
Z = nn_predictions

ax = plt.axes(projection='3d')
# surf = ax.plot_trisurf(x_evaluate[:, 0], x_evaluate[:,1 ], y_evaluate, cmap=cm.jet, linewidth=0.1)
surf = ax.plot_trisurf(evaluate_predictions_x, evaluate_predictions_y, nn_predictions, cmap=cm.jet, linewidth=0.1)

# ax.scatter3D(x_evaluate[:, 0], x_evaluate[:,1 ], y_evaluate, c=y_evaluate, cmap='Greens', label='Testing data');
# ax.scatter3D(evaluate_predictions_input[:, 0], evaluate_predictions_input[:,1 ], nn_predictions, c=nn_predictions,
#                                                                                  cmap='Reds', label='Testing data');

plt.show()

# print(y_evaluate, nn_predictions)
# Disp  lay logs per epoch step
# if (epoch + 1) % display_step == 0:
#     c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
#     print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
#           "W=", sess.run(W), "b=", sess.run(b))


# # Testing example, as requested (Issue #2)
# test_X = numpy.asarray([2])
# test_Y = numpy.asarray([2])
#
# print("Testing... (Mean square loss Comparison)")
# testing_cost = sess.run(
#     tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
#     feed_dict={X: test_X, Y: test_Y})  # same function as cost above
# print("Testing cost=", testing_cost)
# print("Absolute mean square loss difference:", abs(
#     training_cost - testing_cost))
#
