# 2017.12.18
#
# LAB1.
# This lab is about convolutional layer with puppy classifier.
# This lab uses tensorflow for making whole system.
#

# Using gaussian tensor! 

import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import tensorflow as tf
from gaussian_tensor import *



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


	num_class = 2 
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
	W2 = tf.get_variable("W2", [2,2,64,64], initializer = init)
	W3 = tf.get_variable("W3", [2,2,32,8], initializer = init)
	W4 = tf.get_variable("W4", [2,2,8,8], initializer = init)

	#G1 = tf.get_variable("G1", [32], initializer = init)
	#G2 = tf.get_variable("G2", [32], initializer = init)

	G1 = tf.random_normal([1,64], mean = 0.0, stddev = 1.0 , dtype = tf.float32)
	G2 = tf.random_normal([1,32], mean = 0.0, stddev = 1.0,  dtype = tf.float32)



	parameters = {  "W1" : W1,
			"W2" : W2,
			"W3" : W3,
			"W4" : W4,
			"G1" : G1,
			"G2" : G2}


	return parameters 


def extracting_feature(X,parameters, flag = False):


	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3'] 
	W4 = parameters['W4']
	
	G1 = parameters['G1']
	G2 = parameters['G2']

	

	if flag == True:
		
		Z1 = tf.nn.conv2d(X, W1, strides = [1,2,2,1], padding = 'SAME')
        	A1 = tf.nn.relu(Z1)
		
		mean, std = tf.nn.moments(A1, axes=[3])
	        temp1 = tf.multiply(tf.reshape(std,[-1,1]), G1)
		temp1 = tf.reshape(temp1, [-1, np.shape(std)[1], np.shape(std)[2], np.shape(G1)[1]])       		

		temp2 = tf.reshape(mean,[-1,])
       		temp2 = tf.reshape(tf.tile(temp2, [np.shape(G1)[1]]), [-1,np.shape(mean)[1],np.shape(mean)[2],np.shape(G1)[1]])
       		P1 = tf.add(temp2, temp1)


	        Z2 = tf.nn.conv2d(P1, W2, strides = [1,2,2,1], padding = 'SAME')
	        A2 = tf.nn.relu(Z2)
		
		mean, std = tf.nn.moments(A2, axes=[3])
		temp1 = tf.multiply(tf.reshape(std,[-1,1]), tf.reshape(G2,[1,-1]))
		temp1 = tf.reshape(temp1, [-1, np.shape(std)[1], np.shape(std)[1], np.shape(G2)[1]])		

                temp2 = tf.reshape(mean,[-1,])
                temp2 = tf.reshape(tf.tile(temp2, [np.shape(G1)[1]]), [-1,np.shape(mean)[1],np.shape(mean)[2],np.shape(G2)[1]])
                P2 = tf.add(temp2, temp1)


        	Z3 = tf.nn.conv2d(P2, W3, strides = [1,2,2,1], padding = 'SAME')
        	A3 = tf.nn.relu(Z3)
        	#P3 = tf.nn.max_pool(A3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	        Z4 = tf.nn.conv2d(A3, W4, strides = [1,1,1,1], padding = 'SAME')
       		A4 = tf.nn.relu(Z4)
        	#P4 = tf.nn.max_pool(A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


		return Z1,Z2, Z3, Z4

	

	with tf.name_scope("layer1"):
		Z1 = tf.nn.conv2d(X, W1, strides = [1,2,2,1], padding = 'SAME')
		A1 = tf.sigmoid(Z1)
		mean, std = tf.nn.moments(A1, axes=[3]) # mean, std : (N, 100, 100)

	
		##################### Gaussian compression ###########################

		temp1 = tf.multiply(tf.reshape(std,[-1,1]), G1)
		temp1 = tf.reshape(temp1, [-1, np.shape(std)[1], np.shape(std)[2], np.shape(G1)[1]])
	
		temp2 = tf.reshape(mean,[-1,])
		temp2 = tf.reshape(tf.tile(temp2, [np.shape(G1)[1]]), [-1,np.shape(mean)[1],np.shape(mean)[2],np.shape(G1)[1]])
	
		P1 = tf.add(temp2, temp1)
	
		#######################################################################


	with tf.name_scope("layer2"):
        	Z2 = tf.nn.conv2d(P1, W2, strides = [1,2,2,1], padding = 'SAME')
		A2 = tf.sigmoid(Z2)
		mean, std = tf.nn.moments(A2, axes=[3])
	
		temp1 = tf.multiply(tf.reshape(std,[-1,1]), G2)
		temp1 = tf.reshape(temp1, [-1, np.shape(std)[1], np.shape(std)[2], np.shape(G2)[1]])	

		temp2 = tf.reshape(mean,[-1,])
		temp2 = tf.reshape(tf.tile(temp2, [np.shape(G2)[1]]), [-1, np.shape(mean)[1], np.shape(mean)[2], np.shape(G2)[1]])

		P2 = tf.add(temp1,temp2)


	with tf.name_scope("layer3"):

        	Z3 = tf.nn.conv2d(P2, W3, strides = [1,2,2,1], padding = 'SAME')
        	P3 = tf.sigmoid(Z3)

	with tf.name_scope("layer4"):
		Z4 = tf.nn.conv2d(P3, W4, strides = [1,2,2,1], padding = 'SAME')
		A4 = (tf.nn.relu(Z4))


	with tf.name_scope("layer5"):
		P5 = tf.contrib.layers.flatten(A4)
		Z  = tf.contrib.layers.fully_connected(P5, 2, activation_fn = None)



	return Z



def compute_cost(Z, Y):


	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))

	return cost 



def model(learning_rate = 1e-3, num_epochs = 41, print_cost = True):

	
	data = load_dataset(8,1)
	_, n_H, n_W, n_C = np.shape(data['X_train'])


	n_y = np.shape(data['Y_train'])[1]
	param = init_param() 

	tf.summary.histogram("W1",param["W1"])
	tf.summary.histogram("W2",param["W2"])
	tf.summary.histogram("G1",param["G1"])
	tf.summary.histogram("G2",param["G2"])

	X, Y = create_placeholder(n_H, n_W, n_C, n_y)
	norm_X = tf.map_fn(lambda frame : tf.image.per_image_standardization(frame), X)
	

	Z    = extracting_feature(norm_X, param, False)
	K    = extracting_feature(norm_X, param, True)	



	cost = compute_cost(Z, Y)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	init = tf.global_variables_initializer()

	costs = []
		

	with tf.Session() as sess:


		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)

		sess.run(init)
		

		for epoch in range(num_epochs):
			_, temp_cost = sess.run([optimizer, cost], feed_dict = {X:data['X_train'], Y:data['Y_train']})


			if print_cost == True and epoch % 10 == 0 : 
				print("Cost after epoch %i : %f" %(epoch, temp_cost))
				summary = sess.run(merged,feed_dict={X:data['X_train'], Y:data['Y_train']})
				writer.add_summary(summary, epoch)

			if print_cost == True and epoch % 1 == 0:
				costs.append(temp_cost) 


		

		print("========= Test Time ============ ")
		predict_op = tf.argmax(Z, 1)
		correct_prediction = tf.equal(predict_op, tf.argmax(Y,1))

		acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		train_acc = acc.eval({X:data['X_train'], Y:data['Y_train']})
		test_acc  = acc.eval({X:data['X_dev'], Y:data['Y_dev']})

		print("Train acc : ", train_acc)
		print("Test acc : " , test_acc)


		img = np.array(Image.open('./dataset/puppy.jpg').resize((200,200)))
		img = np.expand_dims(img, axis=0)
		Z1,Z2, Z3, Z4 = sess.run(K, feed_dict = {X: img})		
		
		Z1 = np.squeeze(Z1, axis=0)
		Z2 = np.squeeze(Z2, axis=0)
		Z3 = np.squeeze(Z3, axis=0)
		Z4 = np.squeeze(Z4, axis=0)		
		
		'''
		for i in range(32):
	
			plt.subplot(4,8,(i+1))
			stats.probplot(Z1[0,i,:],dist='norm', plot = pylab)


	
		pylab.show()
		'''	
	
		for i in range(24):
			plt.subplot(4,6,i+1)
			plt.matshow(Z1[:,:,i],fignum=False)

		plt.show()
		
		for i in range(24):
			plt.subplot(4,6,i+1)
			#plt.imshow(Z2[:,:,i].astype(np.uint8))
			plt.matshow(Z2[:,:,i], fignum=False)
		plt.show()

		for i in range(32):
			plt.subplot(4,8,i+1)
			#plt.imshow(Z3[:,:,i].astype(np.uint8))
			plt.matshow(Z3[:,:,i], fignum=False)
		
		plt.show()
		
		for i in range(32):
			plt.subplot(4,8,i+1)
			#plt.imshow(Z4[:,:,i].astype(np.uint8))	
			plt.matshow(Z4[:,:,i], fignum=False)

		plt.show()
		#plt.subplot(1,4,3)
		#plt.imshow(Z3.astype(np.uint8))
		#plt.subplot(1,4,4):141
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



