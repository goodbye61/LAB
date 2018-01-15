
# THIS IS MODULE FOR 
# MAKE GENERATE 'GAUSSIAN TENSOR' 
# 

import tensorflow as tf
import numpy as np

def gaussian(activation):

	'''
	Args:
	m : map of mean  (tensor)
	v : map of variance  (tensor)
	
	Return:
	
	gaussian tensor 

	''' 
	_,H, W, C = np.shape(activation)

	iters = C/8
	

	mean, var = tf.nn.moments(activation,axes=[3])
	cons = tf.constant(0.15, tf.float32)

	r = tf.stack([tf.add(mean,tf.multiply(float(t), cons)) for t in range(iters)],axis=-1)
	l = tf.reverse(tf.stack([tf.subtract(mean,tf.multiply(float(t), cons)) for t in range(iters)], axis=-1),[3])
	


	new = tf.concat([l,r],3)


	return new

'''

a = tf.constant([[[1,2,3,4],[4,5,6,4],[7,8,9,10]],[[1,2,3,0],[1,9,9,9],[1,9,9,9]]],dtype = tf.float32)
a = tf.expand_dims(a,axis=0)
b = tf.constant([[5,6,7],[8,9,10],[11,12,13]])

new = gaussian_tensor(a)


with tf.Session() as sess:
	print(" a: ")
	print(sess.run(a))
	print(np.shape(a))

	zz = (sess.run(new))
	print("mean, and var : ")
	print(zz)
	print(np.shape(zz))

'''
