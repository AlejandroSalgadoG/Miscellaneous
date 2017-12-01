from Parameters import *
from Board import *
from Player import *
from Algorithms import *

widthSize = numCol * rectSize
heightSize = numRow * rectSize

boardReady = False
board = None
players = None

def setup():
    global board,players
    size(widthSize, heightSize)
    background(0)
    
    player1 = Player(2, p1Color, p1Start)
    player2 = Player(3, p2Color, p2Start)
    
    players = [player1, player2]
    
    board = Board(players)
    board.drawGrid(255)
    
    board.movePlayer(p1Start, player1)
    board.movePlayer(p2Start, player2)

def draw():
    if boardReady:
        delay(1000)
        executeP1(board, players[0])
        p1turn = False
        
def mouseClicked(): 
    global board
    if not boardReady:
        if mouseButton == LEFT:
            board.createRectangle(mouseX//rectSize, mouseY//rectSize)
        if mouseButton == RIGHT:
            board.removeRectangle(mouseX//rectSize, mouseY//rectSize)
    
def keyPressed():
    global boardReady, board
    boardReady = True
    board.drawGrid(0)