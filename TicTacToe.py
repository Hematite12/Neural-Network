from MonteCarlo import MonteCarloTree
import copy

class TicTacToeTree(MonteCarloTree):
    @staticmethod
    def getInitialState():
        return [[None, None, None],
                [None, None, None],
                [None, None, None]]
    
    @staticmethod
    def getMoves(state, moveColor):
        moves = []
        for row in range(3):
            for col in range(3):
                if state[row][col] == None:
                    moves.append((row, col))
        return moves
    
    @staticmethod
    def stateTransition(state, move, moveColor):
        newState = copy.deepcopy(state)
        newState[move[0]][move[1]] = moveColor
        return newState