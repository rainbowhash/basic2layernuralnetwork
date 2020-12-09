import numpy as np
'''
Simple two layer Neural Network 
'''

# sigmoid activation function
def activation(x):
    return 1/(1+np.exp(-x))


X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
Y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

# initializing a 3*4 array with random wights
syn0 = 2*np.random.random((3, 4))-1
syn1 = 2*np.random.random((4,1))-1
for iter in range(100000):
    layer_0 = X
    # layer 1:input dot weight plus bias of 1 is sent to activation function
    layer_1 = activation(np.dot(X, syn0)+1)
    # layer 2:layer1 dot weight plus bias of 1 is sent to activation function
    layer_2 = activation(np.dot(layer_1, syn1)+1)

    layer_2_error = Y - layer_2
    # error weighted derivative to make sure our sigmoid curve not have a shallow curve
    layer_2_dirv = layer_2_error*(layer_2*(1-layer_2))
    layer_1_error = layer_2_dirv.dot(syn1.T)
    layer_1_dirv = layer_1_error*(layer_1*(1-layer_1))
    syn0 = syn0+layer_0.T.dot(layer_1_dirv)
    syn1 = syn1+layer_1.T.dot(layer_2_dirv)


print(layer_2)