
import numpy as np
import tensorflow as tf 


a = np.random.randint(5, size = (4,4))
b = np.random.randint(5, size = (1,1,4))

print("The value of a : " )
print(a)

print("The value of b : " )
print(b)


print(" = = = = = == = = = = = = =  == = ")

#(1) 
new = np.reshape(a,(16,1)) * np.reshape(b,(1,-1)) # (16,4)
new = np.reshape(new, (4,4,4))



print(new)
print(np.shape(new))
