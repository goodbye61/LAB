
import tensorflow as tf
import numpy as np


a = tf.random_normal([3,3,2], mean =5, stddev = 1.0, dtype=tf.float32)

m,s = tf.nn.moments(a, axes=[2])
m = tf.reshape(m, [-1,])

m = tf.reshape(tf.tile(m, [np.shape(m)[0]]), [3,3,9])

with tf.Session() as sess:
	z= (sess.run(m))
	print(np.shape(z))
