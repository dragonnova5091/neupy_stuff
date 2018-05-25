import numpy as np
import random


x = np.array(([8,4], [5,5], [4,0]), dtype=float)
y = np.array(([96, 50], [85, 32], [72, 10]), dtype=float)

x = x/np.amax(x, axis=0)
y = y/100



class Neural_Net:
    def __init__(self):
        self.input_size = 2
        self.hidden_one = 5
        self.output_size = 2


        self.weights_input = np.random.randn(self.input_size, self.hidden_one)
        self.hidden_one_weights = np.random.randn(self.hidden_one, \
                                                  self.output_size)

        #print(self.weights_input)
        #print(self.hidden_one_weights)

    def sigmoid(self, x):
        return(1/(1+np.exp(-x)))

    def sigmoidPrime(self, s):
        return(s * (1-s))

    def forward(self, x): # x is one set of inputs
        self.z = np.dot(x, self.weights_input)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.hidden_one_weights)
        o = self.sigmoid(self.z3)
        return(o)


    def backward(self, x, y, o):
        #print(str(y))
        self.o_error = y-o
        self.o_delta = self.o_error*self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.hidden_one_weights.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

        self.weights_input += x.T.dot(self.z2_delta)
        self.hidden_one_weights += self.z2.T.dot(self.o_delta)

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)
        return(o)


NN = Neural_Net()

for i in range(0,150000):
    o = NN.train(x, y)

print("output " + str(o))
print("actual values " + str(y))

print(NN.forward(np.array(([2,10]), dtype=float)))
