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
    
    @staticmethod
    def getWinner(state, moveColor):
        for i in range(3):
            row = state[i]
            if row[0]==moveColor and row[1]==moveColor and row[2]==moveColor:
                return moveColor
            if state[0][i]==moveColor and state[1][i]==moveColor and state[2][i]==moveColor:
                return moveColor
        if state[1][1]==moveColor:
            if state[0][0]==moveColor and state[2][2]==moveColor:
                return moveColor
            if state[2][0]==moveColor and state[0][2]==moveColor:
                return moveColor
        isDraw = True
        for i in range(3):
            for j in range(3):
                if state[i][j]==None:
                    isDraw = False
        if isDraw:
            return "draw"
        return None

if __name__ == "__main__":
    t = TicTacToeTree()
    t.printBoard()
    gameIsOver = False
    while not gameIsOver:
        state, gameIsOver = t.decide(10000)
        t.printBoard()

#testing