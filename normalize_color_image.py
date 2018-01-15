from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def create_placeholder(H,W,C):
	
	X = tf.placeholder(tf.float32, [None,H,W,C])

	return X


X_train = [] 

for i in range(8):
	img_name = './dataset/puppy{}.jpg'.format(i+1)
	img = (np.array(Image.open(img_name).resize((200,200))))
	X_train.append(img)





#img = np.array(Image.open('./dataset/puppy.jpg'))

_,h,w,c = np.shape(X_train)
x = create_placeholder(h,w,c)
q = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),x)

with tf.Session() as sess:
	
	new = sess.run(q, feed_dict = {x:X_train})
	print(np.shape(new))
	
	for i in range(np.shape(new)[0]):
		plt.subplot(2,4,i+1)
		plt.imshow(new[i].astype(np.uint8))
	plt.show()




	
