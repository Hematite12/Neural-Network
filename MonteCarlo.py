import math
import random

class Node:
    def __init__(self, color, state, parent=None):
        self.color = color # "w" or "b"
        self.parent = parent
        self.children = []
        self.wins = 0
        self.games = 0
        self.state = state
    
    def isLeaf(self):
        return self.children == []

class MonteCarloTree:
    def __init__(self):
        self.root = Node("w", self.getInitialState())
    
    def getInitialState(self):
        return []
    
    def getScore(self, n):
        return (n.wins/n.games)+math.sqrt(2)*math.sqrt(math.log(self.root.games)/n.games)
    
    def playout(self, n):
        pass
    
    def round(self):
        self.root.games += 1
        currentNode = self.root
        while not currentNode.isLeaf():
            scores = [self.getScore(n) for n in currentNode.children]
            currentNode = currentNode.children[scores.index(max(scores))]
        self.playout(currentNode)