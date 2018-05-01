from MonteCarlo import MonteCarloTree
import copy

class ConnectFourTree(MonteCarloTree):
    @staticmethod
    def getInitialState():
        return [[None for col in range(7)] for row in range(6)]
    
    @staticmethod
    def getMoves(state, moveColor):
        moves = []
        for col in range(7):
            for row in range(5, -1, -1):
                if state[row][col] == None:
                    moves.append((row, col))
                    break
        return moves
    
    @staticmethod
    def stateTransition(state, move, moveColor):
        newState = copy.deepcopy(state)
        newState[move[0]][move[1]] = moveColor
        return newState
    
    @staticmethod
    def getWinner(state, moveColor):
        for row in range(6):
            for col in range(4):
                if state[row][col]==moveColor and state[row][col+1]==moveColor and \
                    state[row][col+2]==moveColor and state[row][col+3]==moveColor:
                    return moveColor
        for col in range(7):
            for row in range(3):
                if state[row][col]==moveColor and state[row+1][col]==moveColor and \
                    state[row+2][col]==moveColor and state[row+3][col]==moveColor:
                    return moveColor
        for row in range(3):
            for col in range(4):
                if state[row][col]==moveColor and state[row+1][col+1]==moveColor and \
                    state[row+2][col+2]==moveColor and state[row+3][col+3]==moveColor:
                    return moveColor
        for row in range(3):
            for col in range(3, 7):
                if state[row][col]==moveColor and state[row+1][col-1]==moveColor and \
                    state[row+2][col-2]==moveColor and state[row+3][col-3]==moveColor:
                    return moveColor
        isDraw = True
        for row in range(6):
            for col in range(7):
                if state[row][col]==None:
                    isDraw = False
        if isDraw:
            return "draw"
        return None
    
    def printBoard(self):
        for row in self.root.state:
            rowString = "|"
            for char in row:
                if char == None: rowString += " "
                elif char: rowString += "X"
                else: rowString += "O"
            rowString.encode("utf-8").decode("ascii")
            print(rowString + "|")
        print()

if __name__ == "__main__":
    t = ConnectFourTree()
    t.printBoard()
    gameIsOver = False
    while not gameIsOver:
        state, gameIsOver = t.decide(5000)
        t.printBoard()