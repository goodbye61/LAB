
import tensorflow as tf 
import keras 
import keras.backend as K
import keras.layers as KL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 


image1 = np.array(Image.open('./data/pup1_mod.jpg'))
print("Sample image size : " , np.shape(image1))

img = tf.placeholder(tf.float32, [None,None, None, None])


up_image = tf.contrib.keras.layers.UpSampling3D(size = (2,2,1))(img)
#inputs = KL.Input(shape = np.shape(image), name = 'input_1')
#up_image = KL.UpSampling2D(size=(2,2) , name = 'up')(inputs)


with tf.Session() as sess:
	
	image = np.expand_dims(image1, axis =0)
	out = sess.run(up_image, feed_dict = {img: image})
	out = np.squeeze(out, axis=0)
	print(np.shape(out))
	print(type(out))
	
	
	plt.subplot(1,2,1)
	plt.imshow(out.astype(np.uint8))
	#plt.show()	

	plt.subplot(1,2,2)
	plt.imshow(image1)
	plt.show()


