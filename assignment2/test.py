import tensorflow as tf
import numpy as np

trainX = np.linspace(-1, 1, 101, dtype=float)
trainX = trainX.reshape(101, 1)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33
print(trainY.dtype)

X = tf.placeholder(tf.float64, shape=[101, 1])
Y = tf.placeholder(tf.float64, shape=[101, 1])

w = tf.Variable(tf.zeros([101, 1],dtype=tf.float64))

y_model = tf.matmul(X, tf.transpose(w))

cost = tf.pow((Y - y_model), 2)

trainop= tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(trainop, feed_dict={X: trainX, Y: trainY})

    print(sess.run(w))

