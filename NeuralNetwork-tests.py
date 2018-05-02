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
    n.trainMultiple(examples, 20000)
    for example in examples:
        output = n.feedForward(example[0])[0]
        toPrint = str(output)
        toPrint.encode("utf-8").decode("ascii")
        print(toPrint)
        if example[1][0] == 0:
            assert(output < .2)
        else:
            assert(output > .8)

def sumTest():
    examples = []
    examples.append(([0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]))
    for i in range(5):
        exIn = [0 for x in range(5)]
        exIn[i] = 1
        examples.append((exIn, [0, 1, 0, 0, 0, 0]))
    for i in range(5):
        for j in range(i+1, 5):
            exIn = [0 for x in range(5)]
            exIn[i] = 1
            exIn[j] = 1
            examples.append((exIn, [0, 0, 1, 0, 0, 0]))
    for i in range(5):
        for j in range(i+1, 5):
            for k in range(j+1, 5):
                exIn = [1 if x==i or x==j or x==k else 0 for x in range(5)]
                examples.append((exIn, [0, 0, 0, 1, 0, 0]))
    for i in range(5):
        for j in range(i+1, 5):
            for k in range(j+1, 5):
                for l in range(k+1, 5):
                    exIn = [1 if x==i or x==j or x==k or x==l else 0 for x in range(5)]
                    examples.append((exIn, [0, 0, 0, 0, 1, 0]))
    n = NeuralNetwork(5, .1, [15, 15, 15, 6])
    n.trainMultiple(examples, 100000)
    for example in examples:
        output = n.feedForward(example[0])
        for i in range(6):
            if example[1][i] == 0:
                assert(output[i] < .2)
            else:
                assert(output[i] > .8)

if __name__ == "__main__":
    tripleXORTest()
    sumTest()