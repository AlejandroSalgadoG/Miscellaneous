from abc import ABC, abstractmethod

from PyQt4.QtGui import *

class Graphic(ABC):

    @abstractmethod
    def draw(self):
        pass

class Window(Graphic):
    def __init__(self):
        self.window = QDialog()
        self.childs = []

    def add(self, child):
        self.childs.append(child)

    def draw(self):
        for child in self.childs:
            child.draw()

        self.window.show()

    def set_title(self, title):
        self.window.setWindowTitle(title)

    def set_size(self, dimensions):
        self.window.setGeometry(*dimensions)

class Button(Graphic):
    def __init__(self, window):
        self.window = window.window
        self.button = QPushButton(self.window)
        self.command = None

    def draw(self):
        self.button.clicked.connect(self.command.execute)

    def set_command(self, command):
        self.command = command

    def set_text(self, text):
        self.button.setText(text)

    def set_size(self, dimensions):
        self.button.move(*dimensions)
