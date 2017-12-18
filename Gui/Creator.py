#!/bin/python

from Graphic import *
from Command import *

from PyQt4.QtGui import *
import sys

from abc import ABC, abstractmethod

class Creator(ABC):
    @abstractmethod
    def create():
        pass

class NewButtonCreator(Creator):
    def create(self, window):
        button = Button(window)
        button.set_text("New")
        button.set_size([70,50])
        button.set_command(NewCommand())
        return button

class WindowCreator(Creator):
    def create(self):
        window = Window()
        window.set_title("Database connection")

        window.set_size([100,100,200,100])
        return window
