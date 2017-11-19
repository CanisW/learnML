# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data 
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
	if not (i/50 - i//50):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print('{0} in 1000:\n accuracy: {1:.2f}\n'.format(i, 
		accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("final accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))