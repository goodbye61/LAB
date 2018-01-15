

import numpy as np
import tensorflow as tf 



a = tf.random_normal([1,1,5], mean = 0.0, stddev =1.0, dtype = tf.float32)
b = tf.random_normal([3,4,5], mean = 0.0, stddev =1.0, dtype = tf.float32)

init = tf.global_variables_initializer() 

m1, s1 = tf.nn.moments(b, axes= [2], keep_dims = True)
exp = tf.tile(m1, [1,1,4])

m2, s2 = tf.nn.moments(b, axes=[2], keep_dims=False)

with tf.Session() as sess:
	
	print(sess.run(m1))
	print(np.shape(m1))
	print(sess.run(m2))
	print(np.shape(m2))

	print(sess.run(exp))
	print(np.shape(exp))
	
