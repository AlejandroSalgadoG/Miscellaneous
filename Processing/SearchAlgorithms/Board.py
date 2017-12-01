from Parameters import rectSize,numCol,numRow
import sys

class Board:
    board = None
    players = None
    
    def __init__(self, players):
        self.board = [[0 for i in range(numCol)] for j in range(numRow)]
        self.players = players
        
    def createRectangle(self, x, y):
        fill(0, 0, 255)
        rect(x*rectSize, y*rectSize, rectSize, rectSize)
        self.board[y][x] = 1
    
    def removeRectangle(self, x, y):
        stroke(255)
        fill(0)
        rect(x*rectSize, y*rectSize, rectSize, rectSize)
        self.board[y][x] = 0
        
    def drawGrid(self, colorFill):
        stroke(colorFill)
        for i in range(1, numCol):
            line(0, i*rectSize, width, i*rectSize)
            line(i*rectSize, 0, i*rectSize, width)
        
    def printBoard(self):
        for i in range(numCol):
            for j in range(numRow):
                sys.stdout.write( str(self.board[i][j]) + " " )
            print
            
    def movePlayer(self, playerPos, player):
        id = player.id - 2
        x,y = player.pos[0], player.pos[1]
        
        stroke(0)
        fill(0)
        ellipseMode(CORNER)
        ellipse(x*rectSize, y*rectSize, rectSize, rectSize)
        self.board[y][x] = 0
        
        player.pos[0], player.pos[1] = playerPos[0], playerPos[1] 
        x,y = playerPos[0], playerPos[1]        
        red = self.players[id].color[0];
        green = self.players[id].color[1];
        blue = self.players[id].color[2];
    
        fill(red, green, blue)
        ellipseMode(CORNER)
        ellipse(x*rectSize, y*rectSize, rectSize, rectSize)
        self.board[y][x] = player.id