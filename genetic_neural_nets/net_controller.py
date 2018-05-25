
import random
import os
import time
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


from neupy import environment
from neupy import algorithms, layers
from neupy import plots
from neupy.estimators import mae

import neupy_tests

#THEANO_FLAGS="device=cuda0"

class NetController:
    def __init__(self):
        self.population = []

        self.size_population = 4

        self.inputneurons = 13
        self.outputneurons = 1
        self.data = datasets.load_boston()

        for i in range(0,self.size_population):
            #connections
            network = layers.join(layers.Input(self.inputneurons))
            num = random.randint(1,2)
            temp1 = list(random.randint(1,50) for i in range(0, num))
            #print(temp1, end="\n\n")
            temp2 = []
            for neu in temp1:
                n = random.randint(1,5)
                temp2.append(n)
                if n == 1:
                    network = network > layers.Sigmoid(neu)
                elif n == 2:
                    network = network > layers.Relu(neu)
                elif n == 3:
                    network = network > layers.Softmax(neu)
                elif n == 4:
                    network = network > layers.Tanh(neu)
                elif n == 5:
                    network = network > layers.LeakyRelu(neu)
                #print(network, end="\n~\n")
            network = network > layers.Sigmoid(self.outputneurons)
            attributes = [temp1, temp2]
            self.population.append([network, attributes, 0]) # 0 --> fitness

        #print(self.population)

        self.run()

    def run(self):
        for index, member in enumerate(self.population):
            print()
            print(member[0], end="\n\n")
            result = neupy_tests.run_neural_net(member[0], self.data)
            print(result)
            self.population[index][2] = [0,result]
        self.fitness()

    def fitness(self):
        for index, member in enumerate(self.population):
            fitness = member[2][1][1] * 100
            print( fitness)
            fitness += member[2][1][0]
            print( fitness)
            self.population[index][2][0] = fitness

        self.population = sorted(self.population, key=lambda member: member[2][0])
        print(self.population)

        best_members = []
        for index in range(0, int(min(10.0, len(self.population)/2.0))):
            best_members.append(self.population[index])

        new_pop = best_members

        for i in range(0, int(self.size_population-(len(self.population)/2.0))):
            new_mem = best_members[random.randint(0,len(best_members)-1)]
            new_mem = self.mutate(new_mem)
            new_pop.append(new_mem)

        print()
        print(self.population)
        print(new_pop)

    def mutate(self, member):
        #[network, [neurons per layer], [tpye of activation function],
        #   [fitness, [time while running, rmsle error]]]

        times = 1
        while random.uniform(0,1) < 0.05:
            times += 1

        for i in range(0,times):

            num = random.randint(1,1)

            if num == 1: # add another layer
                member[1].append(random.randint(0,50))
                member[2].append(random.randint(1,5))
                network = self.init_network(member)
            elif num == 2: #increase size of a layer
                n = random.randint(0,len(member[1]))
                member[1][n] += random.randint(0,20)
                network = self.init_network(member)
            elif num == 3:#decrease layer size
                n = random.randint(0,len(member[1]))
                member[1][n] -= random.randint(-20,0)
                if member[1][n] < 5:
                    member[1][n] = 5
                network = self.init_network(member)
            elif num == 4: #pop layer
                n = random.randint(0,len(member[1]))
                member[1].pop(n)
                member[2].pop(n)
                network = init_network(member)

        network = [network,[member[1]], [member[2]], 0]
        return(network)





    def init_network(self, member):
        network = layers.join(layers.Input(self.inputneurons))
        for index in range(0, len(member[2])):
            if member[2][index] == 1:
                network = network > layers.Sigmoid(member[1][index])
            elif member[2][index] == 2:
                network = network > layers.Relu(member[1][index])
            elif member[2][index] == 3:
                network = network > layers.Softmax(member[1][index])
            elif member[2][index] == 4:
                network = network > layers.Tanh(member[1][index])
            elif member[2][index] == 5:
                network = network > layers.LeakyRelu(member[1][index])
        network = network > layers.Sigmoid(self.outputneurons)
        return(network)



NetController()
