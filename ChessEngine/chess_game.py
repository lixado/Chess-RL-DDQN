from typing import List
from unittest import result

from matplotlib.pyplot import sca

class Piece:
    def __init__(self, piece, position):
        """
            self.piece = ""
            self.position = [0,0]
        """
        self.piece = piece
        self.position = position

class Move:
    def __init__(self, oldPosition, newPosition):
        """
            self.oldPosition = [0, 0]
            self.newPosition = [0, 0]
        """
        self.oldPosition = oldPosition
        self.newPosition = newPosition

    def __eq__(self, other):
        return self.oldPosition == other.oldPosition and self.newPosition == other.newPosition

    def __repr__(self):
        return f'From:({str(self.oldPosition[0])},{str(self.oldPosition[1])}), To:({str(self.newPosition[0])},{str(self.newPosition[1])})'
    
    def __str__(self):
        return f'From:({str(self.oldPosition[0])},{str(self.oldPosition[1])}), To:({str(self.newPosition[0])},{str(self.newPosition[1])})'

class Chess:
    def __init__(self):
        """
            bP = black Pawn
            bR = black Rook

            wP = white Pawn
        """
        # move diagonals
        self.upLeftDiag = [-1, -1]
        self.upRightDiag = [-1, 1]
        self.downRightDiag = [1, 1]
        self.downLeftDiag = [1, -1]
        # move left and right
        self.moveLeft = [0, -1]
        self.moveRight = [0, 1]
        # move up and down
        self.moveUp = [-1, 0]
        self.moveDown = [1, 0]

        self.ResetAndFillBoard()

    def MakeMove(self, move: Move):
        """
            Must be legal moves to not ruin the game, the move must be in GetAvaliableMoves
        """

        if self.board[move.newPosition[0]][move.newPosition[1]] != "__": # if new location is not empty then enemy exists
            if self.currentTurn % 2 == 0: # white playing
                self.blackDead.append(self.board[move.newPosition[0]][move.newPosition[1]])
            else: # black playing
                self.whiteDead.append(self.board[move.newPosition[0]][move.newPosition[1]])

        piece = self.board[move.oldPosition[0]][move.oldPosition[1]] # get the piece
        self.board[move.newPosition[0]][move.newPosition[1]] = piece # update to new location

        self.board[move.oldPosition[0]][move.oldPosition[1]] = "__" # remove old

        self.currentTurn += 1 # move turn 1 time

    
    def GetAvaliableMoves(self, color) -> List[Move]:
        """
            Go trough the avaliable pieces
        """
        possibleMoves = []

        for piece in self.GetPiecesWithPosition(color):
            if "H" in piece.piece: # Horse
                possibleMoves += self.GetHorsePossibleMoves([piece.position[0], piece.position[1]], color)

            if "P" in piece.piece: # Pawns
                possibleMoves += self.GetPawnPossibleMoves([piece.position[0], piece.position[1]], color)

            if "R" in piece.piece: # Rook
                possibleMoves += self.GetRookPossibleMoves([piece.position[0], piece.position[1]], color)

            if "K" in piece.piece: # King
                possibleMoves += self.GetKingPossibleMoves([piece.position[0], piece.position[1]], color)

            if "B" in piece.piece: # Bishop
                possibleMoves += self.GetBishopPossibleMoves([piece.position[0], piece.position[1]], color)

            if "Q" in piece.piece: # Queen
                possibleMoves += self.GetQueenPossibleMoves([piece.position[0], piece.position[1]], color)
        
        return possibleMoves

    def GetQueenPossibleMoves(self, position, color):
        """
            Must be given queen
        """

        return self.GetBishopPossibleMoves(position, color) + self.GetRookPossibleMoves(position, color)


    def GetBishopPossibleMoves(self, startPosition, color):
        """
            Must be given bishop
        """
        possibleMoves = []

        for diag in [self.upLeftDiag, self.upRightDiag, self.downRightDiag, self.downLeftDiag]:
            for i in range(1, 8):
                newPosition = self.MatrixAddition(startPosition, self.MatrixMultiplication(diag, i)) # matrix addition to get new position with multiplication

                result = self.IsEnemy(newPosition, color)
                if result == 0 or result == -2: # if ally or outOfBounds
                    break

                possibleMoves.append(Move(startPosition, newPosition))

                if result == 1: # if enemy
                    break

        return possibleMoves


    def GetKingPossibleMoves(self, startPosition, color):
        """
            Must be given a king
        """ 
        possibleMoves = []

        for diag in [self.upLeftDiag, self.upRightDiag, self.downRightDiag, self.downLeftDiag, self.moveLeft, self.moveRight, self.moveUp, self.moveDown]:
            newPosition = self.MatrixAddition(startPosition, diag) # matrix addition to get new position

            result = self.IsEnemy(newPosition, color)
            if result != 0 and result != -2: # if not ally and not outOfBounds
                possibleMoves.append(Move(startPosition, newPosition))

        """
            check if king is commiting suduku and make ilegal
        """
        # need to check for turn to not have recursion
        enemyMoves = []
        if self.currentTurn % 2 == 0 and color == 'w': # if my turn
            enemyMoves = self.GetAvaliableMoves('b')
        if self.currentTurn % 2 != 0 and color == 'b': 
            enemyMoves = self.GetAvaliableMoves('w')
        
        movesToRemove = []
        for possibleMove in possibleMoves:
            for enemyMove in enemyMoves:
                if 'P' not in self.GetIdFromMoveStart(enemyMove): # if not pawn
                    if possibleMove.newPosition == enemyMove.newPosition:
                        movesToRemove.append(possibleMove)

        # handle the pawns
        enemyPawns = [x.position for x in self.GetPiecesWithPosition(self.GetEnemyColor(color)) if 'P' in x.piece]

        for possibleMove in possibleMoves:
            for pawn in enemyPawns:
                if self.currentTurn % 2 == 0: # white turn
                    downLeft = self.MatrixAddition(pawn, self.downLeftDiag)
                    downRight = self.MatrixAddition(pawn, self.downRightDiag)

                    for diag in [downLeft, downRight]:
                        if possibleMove.newPosition == diag:
                            movesToRemove.append(possibleMove)
                else: # black turn
                    upLeft = self.MatrixAddition(pawn, self.upLeftDiag)
                    upRight = self.MatrixAddition(pawn, self.upRightDiag)

                    for diag in [upLeft, upRight]:
                        if possibleMove.newPosition == diag:
                            movesToRemove.append(possibleMove)


        return [x for x in possibleMoves if x not in movesToRemove]

    def GetRookPossibleMoves(self, startPosition, color):
        """
            Must be given a rook
        """
        possibleMoves = []

        for j in [self.moveLeft, self.moveRight, self.moveUp, self.moveDown]:
            for i in range(1, 8):
                newPosition = self.MatrixAddition(startPosition, self.MatrixMultiplication(j, i)) # matrix addition to get new position with multiplication
                result = self.IsEnemy(newPosition, color)
                if result == -2 or result == 0: # if out of bounds or ally
                    break

                possibleMoves.append(Move(startPosition, newPosition)) # this will run if enemy or empty

                if result == 1: # if enemy cant move more
                    break

        return possibleMoves


    def GetPawnPossibleMoves(self, startPosition, color):
        """
            Must be given a pawn
        """
        possibleMoves = []

        moveUpOrDown = self.moveUp if color == 'w' else self.moveDown
        startingRow = 1 if color == 'b' else 6
        moves = 2 if startPosition[0] == startingRow else 1
        diagonals = [self.upLeftDiag, self.upRightDiag] if color == 'w' else [self.downLeftDiag, self.downRightDiag]

        # check move up
        for i in range(1, moves+1):
            newPosition = self.MatrixAddition(startPosition, self.MatrixMultiplication(moveUpOrDown, i))
            result = self.IsEnemy(newPosition, color)

            if result != -1 : # if not empty square
                break

            possibleMoves.append(Move(startPosition, newPosition)) # this will run if enemy or empty

            if result == 1: # if enemy cant move more
                break
        
        # check diagonals
        for diag in diagonals:
            newPosition = self.MatrixAddition(startPosition, diag) # matrix addition to get new position

            result = self.IsEnemy(newPosition, color)
            if result == 1: # if not ally and not outOfBounds
                possibleMoves.append(Move(startPosition, newPosition))

            
        return possibleMoves


    def GetHorsePossibleMoves(self, startPosition, color):
        """
            Must be given a horse
            Returns [[oldPosition, newPosition], ..]
        """
        possibleMoves = []

        up2 = self.MatrixMultiplication(self.moveUp,2)
        down2 = self.MatrixMultiplication(self.moveDown,2)
        left2 = self.MatrixMultiplication(self.moveLeft,2)
        right2 = self.MatrixMultiplication(self.moveRight,2)
        up2_left1 = self.MatrixAddition(up2,self.moveLeft)
        up2_right1 = self.MatrixAddition(up2,self.moveRight)
        down2_left1 = self.MatrixAddition(down2,self.moveLeft)
        down2_right1 = self.MatrixAddition(down2,self.moveRight)
        left2_up1 = self.MatrixAddition(left2,self.moveUp)
        left2_down1 = self.MatrixAddition(left2,self.moveDown)
        right2_up1 = self.MatrixAddition(right2,self.moveUp)
        right2_down1 = self.MatrixAddition(right2,self.moveDown)

        for j in [up2_left1, up2_right1, down2_left1, down2_right1, left2_up1, left2_down1, right2_up1, right2_down1]:
            newPosition = self.MatrixAddition(startPosition, j) # matrix addition to get new position with multiplication
            result = self.IsEnemy(newPosition, color)
            if result != 0 and result != -2: # if not ally and not outOfBounds
                possibleMoves.append(Move(startPosition, newPosition))

        return possibleMoves

    def GetCurrentTurnPossibleMoves(self) -> List[Move]:
        possibleMoves = []
        if self.currentTurn % 2 == 0:
            possibleMoves = self.GetAvaliableMoves("w")
        else:
            possibleMoves = self.GetAvaliableMoves("b")

        return possibleMoves

    def GetPiecesWithPosition(self, color) -> List[Piece]:
        """
            color:
                w = white
                b = black

            return:
                pieces [[bH, [1,0]], ...]
        """

        pieces = []
        for rowIndex, row in enumerate(self.board):
            for colIndex, element in enumerate(row):
                if(color in element):
                    pieces.append(Piece(element, [rowIndex, colIndex]))

        return pieces

    def IsEnemy(self, position, color):
        """
            returns 
                enemy = 1
                ally = 0 
                empty = -1
                outOfBounds = -2
        """
        # check for out of bounds
        if not (-1 < position[0] < 8 and -1 < position[1] < 8):
            return -2

        if color in self.board[position[0]][position[1]]:
            return 0
        
        if self.board[position[0]][position[1]] == "__":
            return -1

        return 1   

    def GetWinner(self):
        """
            Returns 'w' if white won
                    'b' if black won
                    0 if no winner yet
        """
        if len(self.blackDead) > 0 and self.blackDead[-1] == 'bK':
            return 'w'
        if len(self.whiteDead) > 0 and self.whiteDead[-1] == 'wK':
            return 'b'

        return 0

    def GetEnemyColor(self, color):
        if color == 'w':
            return 'b'
        if color == 'b':
            return 'w'

    def GetIdFromMoveStart(self, move: Move) -> str:
        return self.board[move.oldPosition[0]][move.oldPosition[1]]

    def PrintBoard(self):
        print("Current Board: \n")

        print("↓ " + "  ".join(str(index) for index, _ in enumerate(self.board[0])))
        for rowIndex, row in enumerate(self.board):
            print(f'{str(rowIndex)} {" ".join(str(x) for x in row)}')      

    def MatrixAddition(self, mat1: List, mat2: List) -> List:
        return [i+j for i,j in zip(mat1, mat2)]

    def MatrixMultiplication(self, mat1: List, scalar: int) -> List:
        return [i*scalar for i in mat1]

    def ResetAndFillBoard(self):
        # reset current turn
        self.currentTurn = 0
        self.whiteDead = []
        self.blackDead = []

        # reset board
        self.board = [["bR","bH","bB","bQ","bK","bB","bH","bR"],
                      ["bP","bP","bP","bP","bP","bP","bP","bP"],
                      ["__","__","__","__","__","__","__","__"],
                      ["__","__","__","__","__","__","__","__"],
                      ["__","__","__","__","__","__","__","__"],
                      ["__","__","__","__","__","__","__","__"],
                      ["wP","wP","wP","wP","wP","wP","wP","wP"],
                      ["wR","wH","wB","wQ","wK","wB","wH","wR"]]