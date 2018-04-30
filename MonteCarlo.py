import math
import random
from abc import ABC, abstractmethod

class Node:
    def __init__(self, color, state, moves, parent=None):
        self.color = color # True or False
        self.parent = parent
        self.wins = 0
        self.games = 0
        self.state = state
        self.moves = moves
        self.children = [None for i in moves]
    
    def isLeaf(self):
        return self.children == []
    
    def isExpanded(self):
        for i in self.children:
            if i == None:
                return False
        return True
    
    def expand(self):
        index = self.children.index(None)
        newState = MonteCarloTree.stateTransition(self.state, self.moves[index], not self.color)
        newNode = Node(not self.color, newState, MonteCarloTree.getMoves(newState, not self.color), self)
        self.children[index] = newNode
        return newNode

class MonteCarloTree(ABC):
    def __init__(self):
        initState = self.getInitialState()
        self.root = Node(True, initState, self.getMoves(initState, True))
    
    @staticmethod
    @abstractmethod
    def getInitialState():
        pass
    
    @staticmethod
    @abstractmethod
    def getMoves(state, moveColor):
        pass
    
    @staticmethod
    @abstractmethod
    def stateTransition(state, move, moveColor):
        pass
    
    def playout(self, n):
        pass
    
    def getScore(self, n):
        return (n.wins/n.games)+math.sqrt(2)*math.sqrt(math.log(self.root.games)/n.games)
    
    def iteration(self):
        self.root.games += 1
        currentNode = self.root
        while currentNode.isExpanded():
            scores = [self.getScore(n) for n in currentNode.children]
            currentNode = currentNode.children[scores.index(max(scores))]
        newChild = currentNode.expand()
        self.playout(newChild)