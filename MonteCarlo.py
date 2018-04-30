import math
import random
from abc import ABC, abstractmethod

def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

class Node:
    def __init__(self, color, state, moves, parent=None):
        self.color = color # True or False
        self.parent = parent
        self.wins = 0
        self.games = 0
        self.state = state
        self.moves = moves
        self.children = [None for i in moves]
        self.winner = MonteCarloTree.getWinner(self.state, self.color)
    
    def isLeaf(self):
        return self.children == []
    
    def isTerminal(self):
        return self.winner != None
    
    def isExpanded(self):
        for i in self.children:
            if i == None:
                return False
        return True
    
    def expand(self):
        index = random.choice(indices(self.children, None))
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
    
    @staticmethod
    @abstractmethod
    def getWinner(state, moveColor):
        pass
    
    def backPropagate(self, n):
        winner = n.winner
        while n != None:
            n.games += 1
            if n.color == winner:
                n.wins += 1
            n = n.parent
    
    def playout(self, n):
        while not n.isTerminal():
            n = n.expand()
        self.backPropagate(n)
    
    def getScore(self, n):
        return (n.wins/n.games)+math.sqrt(2)*math.sqrt(math.log(self.root.games)/n.games)
    
    def iteration(self):
        currentNode = self.root
        while currentNode.isExpanded():
            scores = [self.getScore(n) for n in currentNode.children]
            currentNode = currentNode.children[scores.index(max(scores))]
        newChild = currentNode.expand()
        self.playout(newChild)
    
    def chooseMove(self):
        bestChild = self.root.children[0]
        for child in self.root.children[1:]:
            if child.games > bestChild.games:
                bestChild = child
        self.root = bestChild
        return self.root.isTerminal()
    
    def decide(self, numIterations=100):
        for i in range(numIterations):
            self.iteration()
        gameIsOver = self.chooseMove()
        return self.root.state, gameIsOver
    
    def printBoard(self):
        print(self.root.state)
    
    def __str__(self):
        rep = str(self.root.wins) + "/" + str(self.root.games) + "\n"
        for child in self.root.children:
            rep += str(child.wins) + "/" + str(child.games) + " "
        return rep