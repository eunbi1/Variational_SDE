import torch
import matplotlib.pyplot as plt
import numpy as np
import math
X = np.array([[1.,0.], [2.,0.], [3.,0.], [5.,8.],
                  [6.,8.], [7.,8.]])
a = (math.sqrt( 865) - 17)/24
norm = math.sqrt( a**2 +1 )
w = np.array([a, 1])/norm
w = w.reshape(-1,1)

scalar = X@w
scalar = scalar.reshape(1,-1)
trans_X =np.transpose(w*scalar)


x_1 = X[:,0]
y_1 = X[:,1]
x_2 = trans_X[:, 0]
y_2 = trans_X[:, 1]

plt.scatter(x_1,y_1, c ='blue')
plt.scatter(x_2,y_2, c='red')
plt.arrow(0,0, 10*w[0,0], 10*w[1,0])
plt.show()

plt.scatter(scalar[0,:], np.zeros(scalar.shape[1]))

plt.show()