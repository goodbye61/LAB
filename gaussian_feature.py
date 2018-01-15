import pylab
import numpy as np
import scipy.stats as stats

def gaussian_feature(x):

	'''
	x : numpy array (height, width, channel) 
	'''
	_, _, channel = np.shape(x)

	mean_map = np.mean(x, axis=2)
	std_map  = np.std(x, axis=2)
	new = mean_map 

	for i in range(25):
		new = np.dstack((new, mean_map + std_map * 0.1 * (i+1)))
		new = np.dstack((mean_map - std_map * 0.1 * (i+1), new))

	
	return new


	#stats.probplot(new[0,0,:], dist="norm", plot = pylab)
	#pylab.show()




