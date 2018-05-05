import numpy as np
from scipy import signal
import random
import math
import graphics as g

class FCLayer():
    """Fully connected layer; neurons connect to every neuron in the
    previous layer, with randomized weights and biases
    """

    def __init__(self, prevNeurons, numNeurons, actFunc="relu"):
        """FCLayer constructor
        
        Arguments:
            prevNeurons {int} -- the number of neurons in the previous layer
            numNeurons {int} -- the number of neurons in this layer
        
        Keyword Arguments:
            actFunc {str} -- the layer's activation function: "relu", "sigmoid" (default: {"relu"})
        """

        self.prevNeurons = prevNeurons
        self.numNeurons = numNeurons
        self.weights = np.matrix([[random.random()*2-1 for j in range(prevNeurons)] for i in range(numNeurons)])
        self.biases = np.matrix([random.random()*2-1 for i in range(numNeurons)]).transpose()

class ConvLayer():
    """Convolutional layer + activation function, with stride size 1
    """

    def __init__(self, numFeatureMaps=5, windowRows=4, windowCols=4, actFunc="relu"):
        """ConvLayer constructor
        
        Keyword Arguments:
            numFeatureMaps {int} -- number of feature maps, which are each applied to the input image (default: {5})
            windowRows {int} -- number of rows in the sliding feature map window (default: {4})
            windowCols {int} -- number of cols in the sliding feature map window (default: {4})
            actFunc {str} -- the layer's activation function: "relu", "sigmoid" (default: {"relu"})
        """

        self.numFeatureMaps = numFeatureMaps
        self.windowRows = windowRows
        self.windowCols = windowCols
        self.actFunc = actFunc # "sigmoid", "relu"
        self.weights = []
        self.biases = []
        for _ in range(numFeatureMaps):
            self.weights.append(np.matrix([[random.random()*2-1 for i in range(windowRows)] for j in range(windowCols)]))
            self.biases.append(random.random()*2-1)

class PoolingLayer():
    """Pooling layer, shrinks input images based on its activation function
    """

    def __init__(self, windowRows=2, windowCols=2, strideSize=2, actFunc="max"):
        """PoolingLayer constructor
        
        Keyword Arguments:
            windowRows {int} -- number of rows in the sliding pooling window (default: {2})
            windowCols {int} -- number of cols in the sliding pooling window (default: {2})
            strideSize {int} -- number of rows/cols to step the window (default: {2})
            actFunc {str} -- the layer's pooling mode: "max" (default: {"max"})
        """

        self.windowRows = windowRows
        self.windowCols = windowCols
        self.strideSize = strideSize
        self.actFunc = actFunc
        # Selected neurons for max pooling, represented as tuples
        self.forwardSelections = []

class ConvNet():
    """Convolutional neural network. Expected architecture is mixed convolutional and pooling
    layers, followed by some number of fully connected layers. Expected input is a 2D list
    """
    
    def __init__(self, inRows, inCols, learningRate=.1, layerTypes=[], layerInputs=[]):
        """ConvNet constructor
        
        Arguments:
            inRows {int} -- number of rows in the input image
            inCols {int} -- number of cols in the input image
        
        Keyword Arguments:
            learningRate {float} -- the degree to which backpropagation adjusts weights 
            during the gradient descent algorithm (default: {.1})
            layerTypes {list} -- a list of the layer types: "fc" for FCLayer, "c" for ConvLayer,
            "p" for PoolingLayer (default: {[]})
            layerInputs {list} -- a list of parameters for the layers; for ConvLayers
            and PoolingLayers, a tuple of parameters is expected; for FCLayer,
            an int for the number of neurons in that layer (default: {[]})
        """

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
        """Flattens the 2D numpy matrices taken in, and concatenates them"""
        output = inputs[0].flatten()
        for inputMat in inputs[1:]:
            output = np.append(output, inputMat)
        return output
    
    def fullyConnectedOperation(self, layer, inputs):
        """Returns the output of the FCLayer when given the inputs"""
        if layer.actFunc == "relu":
            outputs = self.reluFunc(layer.weights * inputs + layer.biases)
        elif layer.actFunc == "sigmoid":
            outputs = self.sigFunc(layer.weights * inputs + layer.biases)
        return outputs
        
    def layerConvolute(self, layer, inputs):
        """Returns the output of the ConvLayer when given the inputs"""
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
        """Returns the output of the PoolingLayer when given the inputs"""
        if layer.actFunc == "max":
            layer.forwardSelections = []
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
                        chosenRow = -1
                        chosenCol = -1
                        for row in range(layer.windowRows):
                            for col in range(layer.windowCols):
                                newVal = inputMat[overallRow+row, overallCol+col]
                                if newVal > maxVal: 
                                    maxVal = newVal
                                    chosenRow = row
                                    chosenCol = col
                        newMat[newMatRow, newMatCol] = maxVal
                        layer.forwardSelections.append((chosenRow, chosenCol))
                    newMatCol += 1
                newMatRow += 1
            outputs.append(newMat)
        return outputs
    
    def feedForwardHelper(self, inputs):
        """Internal feedForward method, returns a list of the outputs at each layer"""
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
        """Feeds the input image through the network and returns the output as a list
        
        Arguments:
            inputs {list} -- 2D list representing an image
        
        Returns:
            list -- the output of the output layer as a list
        """

        output = self.feedForwardHelper(np.matrix(inputs).transpose())[-1].tolist()
        return [i[0] for i in output]
    
    def maxPoolingError(self, dError, prevLayer, poolLayer):
        """Returns the new dError propagated back from a poolingLayer"""
        newMat = np.zeros(prevLayer.shape)
        loc = 0
        for row in dError.shape[0]:
            for col in dError.shape[1]:
                newMat[poolLayer.forwardSelections[loc][0], poolLayer.forwardSelections[loc][1]] = dError[row, col]
                loc += 1
        return newMat
    
    def backpropagate(self, inputs, expOutputs, outputsL):
        """Adjusts the weights and biases of the network with gradient descent"""
        # Starting error
        dError = expOutputs - outputsL[-1]

        for layerIndex in range(len(self.layers)-1, -1, -1):
            currentLayer = self.layers[layerIndex]
            if type(currentLayer) is FCLayer:
                gradient = np.multiply(dError, self.dsigFunc(outputsL[layerIndex])) * self.learningRate
                if layerIndex-1 >= 0:
                    deltas = gradient * outputsL[layerIndex-1].transpose()
                else:
                    deltas = gradient * inputs.transpose()
                currentLayer.weights += deltas
                currentLayer.biases += gradient
                if layerIndex != 0:
                    dError = currentLayer.weights.transpose() * dError
            elif type(currentLayer) is ConvLayer:
                pass
                if layerIndex != 0:
                    if currentLayer.actFunc == "relu":
                        dError = signal.convolve2d(dError, np.rot90(currentLayer.weights, 2)*self.dreluFunc(outputsL[layerIndex-1]))
                    elif currentLayer.actFunc == "sigmoid":
                        dError = signal.convolve2d(dError, np.rot90(currentLayer.weights, 2)*self.dsigFunc(outputsL[layerIndex-1]))
            elif type(currentLayer) is PoolingLayer:
                if layerIndex != 0:
                    if currentLayer.actFunc == "max":
                        dError = self.maxPoolingError(dError, self.layers[layerIndex-1], currentLayer)
    
    def train(self, inputsL, expOutputsL = []):
        """Train the network on a pair of inputs and expected outputs
        
        Arguments:
            inputsL {list} -- 2D list representing an image
        
        Keyword Arguments:
            expOutputsL {list} -- the expected output of the network.
            If this is not provided, inputsL is assumed to contain
            the inputs as its first element and the expected output
            as its second (default: {[]})
        """

        if expOutputsL != []:
            inputs = np.matrix(inputsL).transpose()
            expOutputs = np.matrix(expOutputsL).transpose()
        else:
            inputs = np.matrix(inputsL[0]).transpose()
            expOutputs = np.matrix(inputsL[1]).transpose()
        outputsL = self.feedForwardHelper(inputs)
        self.backpropagate(inputs, expOutputs, outputsL)
    
    def trainMultiple(self, examples, numTrain):
        """Train the network on random choices of input/output pairs
        in examples, numTrain times
        
        Arguments:
            examples {list} -- a list of input/expected output pairs
            numTrain {int} -- number of times to sample an example randomly
            and train the network on it
        """

        for _ in range(numTrain):
            self.train(random.choice(examples))