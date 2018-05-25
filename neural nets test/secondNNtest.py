


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

this is discontinued

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

'''








#this is a NN test with multiple hidden layers
import time

import numpy as np

import csv

# the speed at which teh network learns
#   a lower rate will learn slower but be more accurate
#   a higher rate learns faster but is less accurate
LEARNRATE = 0.4

with open("Camera.csv", newline='') as csvfile:
    cameras = csv.reader(csvfile)
    data = [r for r in cameras]
#print(data[2])

x_list=[]
y_list=[]
data.pop(0)
data.pop(0)
for line in data:
    text = line[0].split(";")
    #print(text)
    text.pop(0)
    #print(text)
    y = float(text.pop(-1))
    x = text
    #print(y)
    #print(x)

    temp = []
    for i in x:
        #print("~" + str(i) + "~")
        if i != "":
            temp.append(float(i))
    x=temp
    if len(x) == 11:
        x_list.append(x)
        y_list.append([y])

#print( y)
x = np.array(x_list)
y = np.array((y_list))

#print(len(x[0]))
#print(len(y))

class Neural_Net:
    def __init__(self):
        # initialize the size of the NN
        self.input_size = 11
        self.hiddenlayers = [13,13]
        self.output_size = 1

        # weights of the input row
        self.weights_input = np.random.randn(self.input_size, self.hiddenlayers[0])

        # makes a blank array to fill with weights of the hidden layers
        self.hiddenLayersWeights = [] *len(self.hiddenlayers)

        # fills the hiddenLayersWeights with weights that correspond to the row
        for index, layer in enumerate(self.hiddenlayers):
            if index <= len(self.hiddenlayers) -2:
                self.hiddenLayersWeights.append(np.random.randn(layer, self.hiddenlayers[index+1]))
            else :
                self.hiddenLayersWeights.append(np.random.randn(layer, self.output_size))

        # this si the sigmoid function :: the activation function used
        # to modify input
    def sigmoid(self, x, prime=False):
        if prime is False:
            return(1/(1+np.exp(-x)))
        return(x*(1-x))

    def forward(self, x):
        # the list of z values
        self.z_list = []

        # fill the z values
        z = np.dot(x, self.weights_input)
        z2 = self.sigmoid(z)

        # appends teh z values to the array
        self.z_list.append([[z],z2])

        #sets the index ready to call
        index = 0

        # loops through the weights of the hidden layers
        for index, weights in enumerate(self.hiddenLayersWeights):
            #if the set of weights are not for the last row
            if index <= len(self.hiddenLayersWeights) -2:
                # sets the z values and appends them to teh z_list list
                z = np.dot(self.z_list[index][1], weights)
                z2 = self.sigmoid(z)
                index += 1
                self.z_list.append([[z],[z2]])
            else: # if the list is on th last row, then it preps the
                  # output to return, and appends the values to
                  # the list
                z = np.dot(self.z_list[index][1], weights)
                o = self.sigmoid(z)

                self.z_list.append([[z],o])

        return(o)

    # this is the learning function
    # it updates the weights to be closer to desired value
    def backward(self, x, y, o):
        # margin of error for the output
        self.o_error = y-o
        # revert the data to the raw form :: inverse sigmoid * sigmoid
        o_delta = self.o_error*self.sigmoid(o, prime=True)

        # a list to hold the error margin and delta margin
        self.z_errorAndDelta_list = []
        # loop through the range of the hidden layer weights
        for i in range(len(self.hiddenLayersWeights)):
            #print(i)
            i+=1

            #loop through starting at the end going to the front
            if i == 1: # on the first run through
                # the margin of error of the row, using the dot product
                # of the previosu row's raw data, and the weights of the
                # corresponding row
                #print(self.hiddenLayersWeights[-i].T)
                #print(o_delta)
                zx_error = o_delta.dot(self.hiddenLayersWeights[-i].T)
                #zx_error = np.dot(self.hiddenLayersWeights[-i].T, o_delta)

            else: # on teh rest of teh run throughs
                # the margin of error of the row, using the dot product
                # of the previosu row's raw data, and the weights of the
                # corresponding row
                zx_error = zx_delta.dot(self.hiddenLayersWeights[-i].T)


            #the raw data of the row
            #print(self.z_list[-i][1][0])
            zx_delta = zx_error * self.sigmoid(self.z_list[-i][1][0])

            # append the error and data to a list
            self.z_errorAndDelta_list.append([zx_error, zx_delta])

        # weights updates

        #update the input row's weights --> hidden one
        print(x.T, end="\n\n")
        print(self.z_list[0][0])
        self.weights_input += x.T.dot(self.z_list[0][0])

        #loop through each weight of the hidden layers for an update
        for index, weights in enumerate(self.hiddenLayersWeights):
            # update the weights by taking the dot product of the list
            # of the unedited data and the sigmoid product of the data
            # multiplied by the LEARNRATE
            self.hiddenLayersWeights[index] +=  \
                 LEARNRATE * ( np.dot(self.z_list[index+1][0].T, \
                    self.z_list[index+2][1]) )

    def train(self, x, y):
        # makes a list for the output to go into
        o_list = []
        for line in x:
            # gets the output from the network
            o=self.forward(line)
            #appends the output into the output list
            o_list.append(o)

        # changes the output list to a numpy array
        o_list = np.array(o_list)

        # runs the back propagation function on the
        self.backward(x, y, o_list)


        #print(line)

nn = Neural_Net()
nn.train(x,y)
#print(o)
