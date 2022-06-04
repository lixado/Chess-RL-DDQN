from copy import deepcopy
import copy
import tkinter
from tkinter import * 
from tkinter.ttk import *
from PIL import Image, ImageTk
from tkinter import messagebox

class UI:
    def __init__(self, board):
        self.root = tkinter.Tk()
        self.root.title("Chess")
        self.root.attributes('-topmost', True) # always on top
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.done = False

        self.move = [] # [y,x], [y,x]
        self.loading = True
        """
            Load images
        """
        self.imageSize = 55,55
        self.images = self.LoadImages()

        """
            Add widgets
        """
        self.buttons = []
        color = True
        for y, row in enumerate(board):
            
            newRow = []
            for x, element in enumerate(row):
                B = tkinter.Button(self.root, image = self.images[element], bg="#ebecd0" if color else "#779556")
                
                #alternating colors
                if x != 7:
                    color = False if color == True else True

                B.grid(row = y, column = x, sticky = NSEW)
                B.config(command=lambda pos=[y,x]: self.button_pressed(pos))
                newRow.append(B)

            self.buttons.append(newRow)

        self.label = tkinter.Label(self.root, text = "White turn", bg="#FFFFFF", fg="#000000") 
        self.label.grid(row=8, column=0, columnspan=8, sticky=NSEW)

        self.root.update()
        self.loading = False
        self.move = [] 

    def update(self, chess):
        # update turn
        if chess.currentTurn % 2 == 0: # white playing
            self.label.configure(text = "White turn", bg="#FFFFFF", fg="#000000") 
        else: # black playing
            self.label.configure(text = "Black turn", bg="#000000", fg="#FFFFFF") 


        # update board
        for y, row in enumerate(chess.board):
            for x, element in enumerate(row):
                self.buttons[y][x].config(image = self.images[element])

    def resetClick(self):
        # reset move
        self.move = []

        # reset all tiles
        color = True
        for y in range(8):
            for x in range(8):
                self.buttons[y][x].config(bg="#ebecd0" if color else "#779556")
                
                #alternating colors
                if x != 7:
                    color = False if color == True else True


    def drawPossible(self, possibleNextPositions):
        """
            Draw possible next positions given first click
        """
        for element in possibleNextPositions:
            self.buttons[element[0]][element[1]].config(bg = 'blue')

    def button_pressed(self, pos):
        if self.loading == False:
            if len(self.move) < 2: # move already not in action
                self.move.append(pos)
            else:
                print("Button pressed without registering")

    def winner(self, color):
        messagebox.showinfo(title="Winner", message=f'{color} Won')
        self.on_closing()

    def LoadImages(self):
        """
            Returns dict with name: image
        """        
        images = {}
        imageNames = ["wK", "wQ", "wR", "wB", "wH", "wP", "bK", "bQ", "bR", "bB", "bH", "bP", "__"]

        for i in range(len(imageNames)-1):
            image = Image.open("./UI/images/" + imageNames[i] + ".png").resize(self.imageSize, Image.ANTIALIAS)
            images[imageNames[i]] = ImageTk.PhotoImage(image)

        images["__"] = ImageTk.PhotoImage(Image.new('RGBA', self.imageSize, (255, 0, 0, 0)))

        return images

    def on_closing(self):
        self.root.destroy()
        self.done = True


