import numpy as np


x = np.array([ [1,2,3] , [4 ,5 ,6] ,[7,8,9] ])
y= np.array([[1 ,2,2 ]])

# print(x)
# print("")
# #x= x-np.max(x, axis=1)
# print(np.max(x,axis=1).shape)
# f = x - np.max(x,axis=1).reshape((-1,1))
# print(f/np.max(x,axis=1).reshape((-1,1)))
# X_batch= x[np.random.choice(x.shape[0], 2, replace=True)]
# print(x)
# print(X_batch)


X_batch2= x[ind]
print(X_batch2)