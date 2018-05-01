from NeuralNetwork import NeuralNetwork

def tripleXORTest():
    examples = [([0, 0, 0], [0]),
                ([0, 0, 1], [1]),
                ([0, 1, 0], [1]),
                ([1, 0, 0], [1]),
                ([0, 1, 1], [0]),
                ([1, 1, 0], [0]),
                ([1, 0, 1], [0]),
                ([1, 1, 1], [0])]
    n = NeuralNetwork(3, .1, [8, 8, 8, 1])
    n.trainMultiple(examples, 10000)
    for example in examples:
        output = n.feedForward(example[0])[0]
        toPrint = str(output)
        toPrint.encode("utf-8").decode("ascii")
        print(toPrint)
        if example[1][0] == 0:
            assert(output < .2)
        else:
            assert(output > .8)

if __name__ == "__main__":
    tripleXORTest()