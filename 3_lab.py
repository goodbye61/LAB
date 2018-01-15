# 2017.12.18
#
# LAB1.
# This lab is about convolutional layer with puppy classifier.
# This lab uses tensorflow for making whole system.
#

import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import tensorflow as tf
from gaussian_feature import *
from keras import layers as KL


def create_placeholder(n_H, n_W, n_C, n_y):

	X = tf.placeholder(tf.float32, [None, n_H, n_W, n_C])
	Y = tf.placeholder(tf.float32, [None, n_y])
	

	return X, Y 




def load_dataset(m, v):


	data = {}
	X_train = [] 
	X_dev   = [] 

	for i in range(m):
	
		img_name = "./dataset/puppy{}.jpg".format(i+1)
		img = (np.array(Image.open(img_name).resize((200,200))))	
		X_train.append(img)
				

	for i in range(v):

		img_name = './dataset/cat{}.jpg'.format(i+1)
		img = (np.array(Image.open(img_name).resize((200,200))))
		X_dev.append(img)


	label = np.loadtxt('./dataset/y_train.txt', unpack= True, dtype= 'int32')


	Y_train = label[0:m]
	Y_dev   = np.transpose(label[-(v):,])


	num_class = 3
	Y_train = np.eye(num_class)[Y_train]
	Y_dev   = np.eye(num_class)[Y_dev]

	

	data['X_train'] = X_train
	data['X_dev']   = X_dev
	data['Y_train'] = Y_train
	data['Y_dev']   = Y_dev 


	return data 



def init_param():
	
	init = tf.contrib.layers.xavier_initializer()

	W1 = tf.get_variable("W1", [2,2,3,256], initializer = init)
	W2 = tf.get_variable("W2", [2,2,256,64], initializer = init)
	W3 = tf.get_variable("W3", [2,2,64,16], initializer = init)
	W4 = tf.get_variable("W4", [2,2,16,16], initializer = init)

	Q1 = tf.get_variable("Q1", [3,3,256,64], initializer = init)


	parameters = {  "W1" : W1,
			"W2" : W2,
			"W3" : W3,
			"W4" : W4,
			"Q1" : Q1}


	return parameters 


def extracting_feature(X,parameters, flag = False):


	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3'] 
	W4 = parameters['W4']
	
	Q1= parameters['Q1']

	if flag == True:
		
		Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
        	A1 = tf.nn.relu(Z1)
	        P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

		T = KL.UpSampling2D(size=(2,2))(P1)
		T_1 = tf.nn.conv2d(T, Q1, strides =[1,2,2,1], padding ='SAME')
	
	        Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
	        A2 = tf.nn.relu(Z2)
		P2 = tf.add(T_1, Z2)

	        P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

        	Z3 = tf.nn.conv2d(P2, W3, strides = [1,2,2,1], padding = 'SAME')
        	A3 = tf.nn.relu(Z3)
        	P3 = tf.nn.max_pool(A3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	        Z4 = tf.nn.conv2d(P3, W4, strides = [1,2,2,1], padding = 'SAME')
       		A4 = tf.nn.relu(Z4)
        	P4 = tf.nn.max_pool(A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


		return Z1, P1, Z2, P2 , Z3, P3 
		




	Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
	A1 = tf.nn.relu(Z1)
	P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	# Most shallow feature map : P1 

	T = KL.UpSampling2D(size = (2,2))(P1)
	T_1 = tf.nn.conv2d(T, Q1, strides = [1,2,2,1], padding = 'SAME')


        Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
        A2 = tf.nn.relu(Z2)
	P2 = tf.add(T_1, Z2)	

	P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	print("P2 size : ", P2)

        Z3 = tf.nn.conv2d(P2, W3, strides = [1,2,2,1], padding = 'SAME')
        A3 = tf.nn.relu(Z3)
        P3 = tf.nn.max_pool(A3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	print("P3 szie : ", P3)

	Z4 = tf.nn.conv2d(P3, W4, strides = [1,2,2,1], padding = 'SAME')
        A4 = tf.nn.relu(Z4)
        P4 = tf.nn.max_pool(A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


	P5 = tf.contrib.layers.flatten(P4)
	Z  = tf.contrib.layers.fully_connected(P5, 3, activation_fn = None)



	return Z



def compute_cost(Z, Y):


	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))

	return cost 



def model(learning_rate = 1e-3, num_epochs = 41, print_cost = True):

	
	data = load_dataset(16,1)
	_, n_H, n_W, n_C = np.shape(data['X_train'])


	n_y = np.shape(data['Y_train'])[1]
	param = init_param() 

	X, Y = create_placeholder(n_H, n_W, n_C, n_y)
	norm_X = tf.map_fn(lambda frame : tf.image.per_image_standardization(frame), X)


	Z    = extracting_feature(norm_X, param, False)
	K    = extracting_feature(norm_X, param, True)	



	cost = compute_cost(Z, Y)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	init = tf.global_variables_initializer()

	costs = []
		

	with tf.Session() as sess:
		sess.run(init)
		

		for epoch in range(num_epochs):
			_, temp_cost = sess.run([optimizer, cost], feed_dict = {X:data['X_train'], Y:data['Y_train']})



			if print_cost == True and epoch % 20 == 0 : 
				print("Cost after epoch %i : %f" %(epoch, temp_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(temp_cost) 


		

		print("========= Test Time ============ ")
		predict_op = tf.argmax(Z, 1)
		correct_prediction = tf.equal(predict_op, tf.argmax(Y,1))

		acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		train_acc = acc.eval({X:data['X_train'], Y:data['Y_train']})
		test_acc  = acc.eval({X:data['X_dev'], Y:data['Y_dev']})

		print("Train acC : ", train_acc)
		print("Test acc : " , test_acc)


		img = np.array(Image.open('./dataset/puppy.jpg').resize((200,200)))
		img = np.expand_dims(img, axis=0)
		Z1, P1, Z2, P2, Z3, P3 = sess.run(K, feed_dict = {X: img})		
		
		Z1 = np.squeeze(Z1, axis=0)
		P1 = np.squeeze(P1, axis=0)

		Z2 = np.squeeze(Z2, axis=0)
		P2 = np.squeeze(P2, axis=0)
		
		Z3 = np.squeeze(Z3, axis=0)
		P3 = np.squeeze(P3, axis=0)
		

		#Z4 = np.squeeze(Z4, axis=0)		

		'''
		for i in range(32):
		
			plt.subplot(4,8,(i+1))
			stats.probplot(Z1[0,i,:],dist='norm', plot = pylab)
	
		pylab.show()
		time.sleep(5)
		''' 
	
		#plt.subplot(1,4,1)
		#plt.imshow(a.astype(np.uint8))
		
		'''
		for i in range(64):
			plt.subplot(8,8,i+1)
			plt.matshow(Z1[:,:,i],fignum=False)

		plt.show()
		
		for i in range(64):
			plt.subplot(8,8,i+1)
			#plt.imshow(Z2[:,:,i].astype(np.uint8))
			plt.matshow(P1[:,:,i], fignum=False)
		plt.show()
		'''
		for i in range(32):
			plt.subplot(8,4,i+1)
			#plt.imshow(Z3[:,:,i].astype(np.uint8))
			plt.matshow(Z2[:,:,i], fignum=False)
		
		plt.show()
		
		for i in range(32):
			plt.subplot(8,4,i+1)
			#plt.imshow(Z4[:,:,i].astype(np.uint8))	
			plt.matshow(P2[:,:,i], fignum=False)

		plt.show()


		for i in range(16):
			plt.subplot(4,4,i+1)
			plt.matshow(Z3[:,:,i], fignum=False)
		
		plt.show()
		
		for i in range(16):
			plt.subplot(4,4,i+1)
			plt.matshow(P3[:,:,i], fignum=False)

		plt.show()


		#plt.subplot(1,4,3)
		#plt.imshow(Z3.astype(np.uint8))
		#plt.subplot(1,4,4)
		#plt.imshow(Z4.astype(np.uint8))

		#plt.show()



#data = load_dataset(7,1)
#print((np.shape(data['X_train'])))

model()














 
	




'''
# Image loading 
data = []

image1 = Image.open('puppy1.jpg')
image2 = Image.open('puppy4.jpg')
image3 = Image.open('puppy5.jpg')

data.append(image1)
data.append(image2)
data.append(image3)


# Image esize (Unifying the size of image)


l = len(data)

for i in range(l):
        data[i] = np.array(data[i].resize((200,200)))



print(np.shape(data))
'''
















	
	




















 















# Checking the size 

#for i in range(l) :
#	print(data[i].size)
#	data[i].show()


























	
