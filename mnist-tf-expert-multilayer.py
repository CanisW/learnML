# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data 
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

def W_var(shape):
	init = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init)
	
def b_var(shape):
	init = tf.constant(0.1, shape=shape)
	return tf.Variable(init)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
	
def max_poll_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	
#layer conv 1
W_conv1 = W_var([5,5,1,32])
b_conv1 = b_var([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_poll_2x2(h_conv1)

#layer conv 2
W_conv2 = W_var([5,5,32,64])
b_conv2 = b_var([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_poll_2x2(h_conv2)

#layer FC 1
W_fc1 = W_var([7*7*64, 1024])
b_fc1 = b_var([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#layer FC 2
W_fc2 = W_var([1024, 10])
b_fc2 = b_var([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#training
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_predi = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accur = tf.reduce_mean(tf.cast(correct_predi, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i % 100 == 0 or i == 19999:
			train_accur = accur.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
			print('step: {0}, \ntraining accuracy: {1:.2f}\n'.format(i, train_accur))
		train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
	print('test accuracy: {0}'.format(accur.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})))
	
