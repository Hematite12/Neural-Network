import numpy as np
import random
import math

class Layer():
    def __init__(self, type, prevNeurons, numNeurons):
        self.type = type
        self.prevNeurons = prevNeurons
        self.numNeurons = numNeurons
        self.weights = np.matrix([[random.random()-.5 for j in range(prevNeurons)] for i in range(numNeurons)])
        self.biases = np.matrix([random.random()-.5 for i in range(numNeurons)]).transpose()
        print(self.weights)

class NeuralNetwork():
    def __init__(self, inputSize, learningRate, neurL):
        self.inputSize = inputSize
        self.learningRate = learningRate
        self.layers = [Layer("standard", inputSize, neurL[0])]
        for i in range(1, len(neurL)):
            self.layers.append(Layer("standard", neurL[i-1], neurL[i]))
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def dsigmoid(x):
        return x * (1-x)
    
    def feedForwardHelper(self, inputs):
        actFunc = np.vectorize(NeuralNetwork.sigmoid)
        currentOutput = inputs
        outputsL = []
        for layer in self.layers:
            currentOutput = actFunc(layer.weights * currentOutput + layer.biases)
            outputsL.append(currentOutput)
        return outputsL
    
    def feedForward(self, inputs):
        return self.feedForwardHelper(np.matrix(inputs).transpose())[-1]
    
    def train(self, inputsL, expOutputsL = []):
        if expOutputsL != []:
            inputs = np.matrix(inputsL).transpose()
            expOutputs = np.matrix(expOutputsL).transpose()
        else:
            inputs = np.matrix(inputsL[0]).transpose()
            expOutputs = np.matrix(inputsL[1]).transpose()
        dsigFunc = np.vectorize(NeuralNetwork.dsigmoid)
        
        outputsL = self.feedForwardHelper(inputs)
        error = expOutputs - outputsL[-1]

        for layerIndex in range(len(self.layers)-1, -1, -1):
            gradient = self.learningRate * np.multiply(error, dsigFunc(outputsL[layerIndex]))
            deltas = gradient * outputsL[layerIndex-1].transpose()
            self.layers[layerIndex].weights += deltas
            self.layers[layerIndex].biases += gradient
            error = self.layers[layerIndex].weights.transpose() * error

if __name__ == "__main__":
    n = NeuralNetwork(2, .02, [4, 1])
    examples = [([0, 0], [0]),
                ([0, 1], [1]),
                ([1, 0], [1]),
                ([1, 1], [0])]
    for i in range(3000):
        n.train(random.choice(examples))
    print(n.feedForward([0, 0]))
    print(n.feedForward([0, 1]))
    print(n.feedForward([1, 0]))
    print(n.feedForward([1, 1]))