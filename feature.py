
import tensorflow as tf
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

# Image loading 

img = np.array(Image.open('puppy.jpg'))
X = tf.placeholder(tf.float32, [None,None, None, None])

ini = tf.contrib.layers.xavier_initializer()

W_a = tf.get_variable("W_a", [2,2,3,256], initializer = ini)
W_b = tf.get_variable("W_b", [2,2,3,3], initializer = ini)

Z_a = tf.nn.conv2d(X, W_a, strides = [1,1,1,1], padding = 'SAME')
Z_b = tf.nn.conv2d(X, W_b, strides = [1,1,1,1], padding = 'SAME')

init = tf.global_variables_initializer()


with tf.Session() as sess:
	sess.run(init)
	
	img = np.expand_dims(img, axis=0)
	r1 = sess.run(Z_a, feed_dict = {X:img})
	r2 = sess.run(Z_b, feed_dict = {X:img})
	
	print(np.shape(r2))
	r2 = np.squeeze(r2,axis=0)	

	plt.imshow(r2.astype(np.uint8))
	plt.show()

