

import tensorflow as tf
import numpy as np 

'''
Make custom gradient
When adding new ops in Tensorflow, you must use tf.RegisterGradeint
to register a gradient function which computes gradients with respect to
the op's input tensors given gradient with respec to the ops' output
tensors. 

''' 


@tf.RegisterGradient("QuantizeGrad")

def quantize_grad(op,x):
	pass

G = tf.get_default_graph()
def quantize(x):
	with G.gradient_override_map({"Sign": "QuantizeGrad"}):
		E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
		return tf.sign(x/E) * E 





with tf.Session() as sess: 














