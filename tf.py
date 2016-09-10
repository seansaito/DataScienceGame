import tensorflow as tf
import numpy as np

from sklearn.cross_validation import train_test_split

from data_iterator import SequentialIterator

# Load the data here

# Params
x_dim = 1000
batchsize = 500

x_ph = tf.placeholder(tf.float32, [None, x_dim])
y_ph = tf.placeholder(tf.float32, [None, 1])

interm_dimension = 20

W_1 = tf.Variable(tf.random_normal[x_dim, interm_dimension])
b_1 = tf.Variable(tf.random_normal[interm_dimension])

interm_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal[interm_dimension, 1])
b_2 = tf.Variable(tf.random_normal[interm_dimension])

y = tf.sigmoid(tf.matmul(tf.matmul(interm_1, W_2) + b_2))

mse = tf.reduce_mean((y_ph - y) ** 2)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)
