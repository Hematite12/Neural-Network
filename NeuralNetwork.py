import numpy as np
import random
import math

class Layer():
    def __init__(self, type, prevNeurons, numNeurons, weights):
        self.type = type
        self.prevNeurons = prevNeurons
        self.numNeurons = numNeurons
        self.weights = np.matrix([[random.random() for j in range(prevNeurons)] for i in range(numNeurons)])
        self.biases = np.matrix([random.random() for i in range(numNeurons)]).transpose()

class NeuralNetwork():
    def __init__(self, inputSize, outputSize, numHidden, hiddenSize, learningRate):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numHidden = numHidden
        self.hiddenSize = hiddenSize
        self.learningRate = learningRate
        self.outputLayer = np.matrix([[random.random() for j in range(hiddenSize)] for i in range(outputSize)])
        firstHiddenLayer = [np.matrix([[random.random() for j in range(inputSize)] for i in range(hiddenSize)])]
        subsequentHiddenLayers = [np.matrix([[random.random() for j in range(hiddenSize)] for k in range(hiddenSize)]) for i in range(1, numHidden)]
        self.hiddenLayers = firstHiddenLayer + subsequentHiddenLayers
        outputBias = [np.matrix([random.random() for j in range(outputSize)]).transpose()]
        hiddenBiases = [np.matrix([random.random() for j in range(hiddenSize)]).transpose() for i in range(1, numHidden)]
        self.biases = outputBias + hiddenBiases
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def dsigmoid(x):
        return x * (1-x)
    
    def feedForwardHelper(self, inputs):
        actFunc = np.vectorize(NeuralNetwork.sigmoid)
        outputs = inputs
        outputsL = []
        for layer in self.hiddenLayers + [self.outputLayer]:
            outputs = actFunc(layer * outputs)
            outputsL.append(outputs)
        return outputsL
    
    def feedForward(self, inputs):
        return self.feedForwardHelper(np.matrix(inputs).transpose())[-1]
    
    def train(self, inputs, expOutputs):
        inputs = np.matrix(inputs).transpose()
        expOutputs = np.matrix(expOutputs).transpose()
        
        outputsL = self.feedForwardHelper(inputs)
        outputError = expOutputs - outputsL[-1]
        dsigFunc = np.vectorize(NeuralNetwork.dsigmoid)

        gradient = self.learningRate * outputError * dsigFunc(outputsL[-1])
        deltas = gradient * outputsL[-2].transpose()
        self.outputLayer += deltas
        self.biases[-1] += gradient

        prevLayer = self.outputLayer
        prevError = outputError
        for i in range(self.numHidden):
            prevError = prevLayer.transpose() * prevError

            gradient = self.learningRate * prevError * dsigFunc(outputsL[-i-1])
            deltas = gradient * outputsL[-i-2].transpose()
            self.hiddenLayers[-i-1] += deltas
            print(self.biases[-1-1])
            print(gradient)
            self.biases[-i-1] += gradient

            prevLayer = self.hiddenLayers[-i-1]

if __name__ == "__main__":
    n = NeuralNetwork(2, 1, 3, 4, .01)
    n.train([0, 1], [1])