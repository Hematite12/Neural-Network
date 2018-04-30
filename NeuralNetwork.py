import numpy as np
import random
import math
import graphics as g

class Layer():
    def __init__(self, type, prevNeurons, numNeurons):
        self.type = type
        self.prevNeurons = prevNeurons
        self.numNeurons = numNeurons
        self.weights = np.matrix([[random.random()*2-1 for j in range(prevNeurons)] for i in range(numNeurons)])
        self.biases = np.matrix([random.random()*2-1 for i in range(numNeurons)]).transpose()

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
            if layerIndex != len(self.layers)-1:
                error = self.layers[layerIndex+1].weights.transpose() * error
            gradient = np.multiply(error, dsigFunc(outputsL[layerIndex])) * self.learningRate
            if layerIndex-1 >= 0:
                deltas = gradient * outputsL[layerIndex-1].transpose()
            else:
                deltas = gradient * inputs.transpose()
            self.layers[layerIndex].weights += deltas
            self.layers[layerIndex].biases += gradient
    
    def show(self, NR = 25):
        win = g.GraphWin("Neural Network", 800, 400)
        yPos = 2*NR
        for i in range(self.inputSize):
            n = g.Circle(g.Point(2*NR, yPos), NR)
            n.draw(win)
            yPos += 4*NR
        yPos = 2*NR
        xPos = 12*NR
        for layer in self.layers:
            for i in range(layer.numNeurons):
                n = g.Circle(g.Point(xPos, yPos), NR)
                n.draw(win)
                prevX = xPos - 10*NR
                prevY = 2*NR
                for j in range(layer.prevNeurons):
                    l = g.Line(g.Point(prevX+NR, prevY), g.Point(xPos-NR, yPos))
                    l.draw(win)
                    textLoc = g.Point((prevX+xPos)/2, (prevY + yPos)/2)
                    t = g.Text(textLoc, str(round(layer.weights.item((i, j)), 3)))
                    t.draw(win)
                    prevY += 4*NR
                biasText = g.Text(g.Point(xPos, yPos), str(round(layer.biases.item(i), 3)))
                biasText.draw(win)
                yPos += 4*NR
            yPos = 2*NR
            xPos += 10*NR
        win.getMouse()
        win.close()

if __name__ == "__main__":
    n = NeuralNetwork(2, .1, [4, 1])
    examples = [([0, 0], [0]),
                ([0, 1], [1]),
                ([1, 0], [1]),
                ([1, 1], [0])]
    for x in range(1):
        for i in range(10000):
            n.train(random.choice(examples))
        print(n.feedForward([0, 0]))
        print(n.feedForward([0, 1]))
        print(n.feedForward([1, 0]))
        print(n.feedForward([1, 1]))
        n.show(25)