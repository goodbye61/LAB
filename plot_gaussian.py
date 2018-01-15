
import numpy as np
import matplotlib.pyplot as plt 


eps = np.random.normal(0.0,1.0, 10000)
var = 1000

eps_2 = eps * var 

plt.subplot(1,2,1)
plt.hist(eps,100)

plt.subplot(1,2,2)
plt.hist(eps_2, 100)

plt.show()


print(" mean of eps : " , np.mean(eps), "std : ", np.std(eps))
print(" mean of eps_2 : ", np.mean(eps_2), "std : " , np.std(eps_2))



