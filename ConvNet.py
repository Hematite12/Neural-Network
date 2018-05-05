import numpy as np
from scipy import signal
import random
import math
import graphics as g

class FCLayer():
    def __init__(self, prevNeurons, numNeurons, actFunc="relu"):
        self.prevNeurons = prevNeurons
        self.numNeurons = numNeurons
        self.weights = np.matrix([[random.random()*2-1 for j in range(prevNeurons)] for i in range(numNeurons)])
        self.biases = np.matrix([random.random()*2-1 for i in range(numNeurons)]).transpose()

class ConvLayer():
    def __init__(self, numFeatureMaps=5, windowRows=4, windowCols=4, actFunc="relu"):
        self.numFeatureMaps = numFeatureMaps
        self.windowRows = windowRows
        self.windowCols = windowCols
        self.actFunc = actFunc # "sigmoid", "relu"
        self.weights = []
        self.biases = []
        for i in range(numFeatureMaps):
            self.weights.append(np.matrix([[random.random()*2-1 for i in range(windowRows)] for j in range(windowCols)]))
            self.biases.append(random.random()*2-1)

class PoolingLayer():
    def __init__(self, windowRows=2, windowCols=2, strideSize=2, actFunc="max"):
        self.windowRows = windowRows
        self.windowCols = windowCols
        self.strideSize = strideSize
        self.actFunc = actFunc

class ConvNet():
    def __init__(self, inRows, inCols, learningRate=.1, layerTypes=[], layerInputs=[]):
        self.sigFunc = np.vectorize(ConvNet.sigmoid)
        self.dsigFunc = np.vectorize(ConvNet.dsigmoid)
        self.reluFunc = np.vectorize(ConvNet.relu)
        self.dreluFunc = np.vectorize(ConvNet.drelu)
        self.inRows = inRows
        self.inCols = inCols
        self.learningRate = learningRate
        self.layers = []
        inputIsPicture = True
        currentInputRows = inRows
        currentInputCols = inCols
        currentFeatureMaps = 1
        for i in range(len(layerTypes)):
            if layerTypes[i] == "c":
                if layerInputs[i] == None:
                    newLayer = ConvLayer()
                else:
                    newLayer = ConvLayer(*layerInputs[i])
                self.layers.append(newLayer)
                currentFeatureMaps *= newLayer.numFeatureMaps
                currentInputRows += -newLayer.windowRows + 1
                currentInputCols += -newLayer.windowCols + 1
            elif layerTypes[i] == "p":
                if layerInputs[i] == None:
                    newLayer = PoolingLayer()
                else:
                    newLayer = PoolingLayer(*layerInputs[i])
                self.layers.append(newLayer)
                currentInputRows = math.ceil(currentInputRows/newLayer.strideSize)
                currentInputCols = math.ceil(currentInputCols/newLayer.strideSize)
            elif layerTypes[i] == "fc":
                if inputIsPicture:
                    inputIsPicture = False
                    inputNeurons = currentFeatureMaps * currentInputRows * currentInputCols
                else:
                    inputNeurons = self.layers[-1].numNeurons
                self.layers.append(FCLayer(inputNeurons, layerInputs[i]))
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def dsigmoid(x):
        return x * (1-x)
    
    @staticmethod
    def relu(x):
        return max(0, x)
    
    @staticmethod
    def drelu(x):
        if x > 0: return 1
        else: return 0
    
    @staticmethod
    def flatten(inputs):
        output = inputs[0].flatten()
        for inputMat in inputs[1:]:
            output = np.append(output, inputMat)
        return output
    
    def fullyConnectedOperation(self, layer, inputs):
        if layer.actFunc == "relu":
            outputs = self.reluFunc(layer.weights * inputs + layer.biases)
        elif layer.actFunc == "sigmoid":
            outputs = self.sigFunc(layer.weights * inputs + layer.biases)
        return outputs
        
    def layerConvolute(self, layer, inputs):
        outputs = []
        for inputMat in inputs:
            for i in range(layer.numFeatureMaps):
                if layer.actFunc == "relu":
                    newMat = self.reluFunc(signal.convolve2d(layer.weights[i], inputMat, "valid") + layer.biases[i])
                elif layer.actFunc == "sigmoid":
                    newMat = self.sigFunc(signal.convolve2d(layer.weights[i], inputMat, "valid") + layer.biases[i])
                outputs.append(newMat)
        return outputs
    
    def pool(self, layer, inputs):
        outputs = []
        for inputMat in inputs:
            inputRows = inputMat.shape[0]
            inputCols = inputMat.shape[1]
            newMat = np.empty((math.ceil(inputRows/layer.strideSize), math.ceil(inputCols/layer.strideSize)))
            newMatRow = 0
            for overallRow in range(0, inputRows, layer.strideSize):
                newMatCol = 0
                for overallCol in range(0, inputCols, layer.strideSize):
                    if layer.actFunc == "max":
                        maxVal = -1
                        for row in range(layer.windowRows):
                            for col in range(layer.windowCols):
                                newVal = inputMat[overallRow+row, overallCol+col]
                                if newVal > maxVal: 
                                    maxVal = newVal
                        newMat[newMatRow, newMatCol] = maxVal
                    newMatCol += 1
                newMatRow += 1
            outputs.append(newMat)
        return outputs
    
    def feedForwardHelper(self, inputs):
        currentInputIsPicture = True
        outputsL = []
        for layer in self.layers:
            if currentInputIsPicture and type(layer) is FCLayer:
                inputs = ConvNet.flatten(inputs)
                currentInputIsPicture = False
            if type(layer) is FCLayer:
                inputs = self.fullyConnectedOperation(layer, inputs)
            elif type(layer) is ConvLayer:
                inputs = self.layerConvolute(layer, inputs)
            elif type(layer) is PoolingLayer:
                inputs = self.pool(layer, inputs)
            outputsL.append(inputs)
        return outputsL
    
    def feedForward(self, inputs):
        output = self.feedForwardHelper(np.matrix(inputs).transpose())[-1].tolist()
        return [i[0] for i in output]
    
    def backpropagate(self, inputs, expOutputs, outputsL):
        dError = expOutputs - outputsL[-1]

        for layerIndex in range(len(self.layers)-1, -1, -1):
            if layerIndex != len(self.layers)-1:
                if type(self.layers[layerIndex]) is FCLayer:
                    dError = self.layers[layerIndex+1].weights.transpose() * dError
                elif type(self.layers[layerIndex]) is ConvLayer:
                    if self.layers[layerIndex].actFunc == "relu":
                        dError = signal.convolve2d(dError, np.rot90(self.layers[layerIndex+1].weights, 2)*self.dreluFunc(outputsL[layerIndex]))
                    elif self.layers[layerIndex].actFunc == "sigmoid":
                        dError = signal.convolve2d(dError, np.rot90(self.layers[layerIndex+1].weights, 2)*self.dsigFunc(outputsL[layerIndex]))
                elif type(self.layers[layerIndex]) is PoolingLayer:
                    pass
            gradient = np.multiply(dError, self.dsigFunc(outputsL[layerIndex])) * self.learningRate
            if layerIndex-1 >= 0:
                deltas = gradient * outputsL[layerIndex-1].transpose()
            else:
                deltas = gradient * inputs.transpose()
            self.layers[layerIndex].weights += deltas
            self.layers[layerIndex].biases += gradient
    
    def train(self, inputsL, expOutputsL = []):
        if expOutputsL != []:
            inputs = np.matrix(inputsL).transpose()
            expOutputs = np.matrix(expOutputsL).transpose()
        else:
            inputs = np.matrix(inputsL[0]).transpose()
            expOutputs = np.matrix(inputsL[1]).transpose()
        outputsL = self.feedForwardHelper(inputs)
        self.backpropagate(inputs, expOutputs, outputsL)
    
    def trainMultiple(self, examples, numTrain):
        for i in range(numTrain):
            self.train(random.choice(examples))