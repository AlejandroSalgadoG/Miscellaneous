#!/bin/python

from abc import ABC, abstractmethod

class Instrument(ABC): # Define class as abstract

    @abstractmethod # Define abstract method
    def play(self):
        pass

class Guitar(Instrument): # Inherit from Instrument

    def play(self): # Implement abstract method
        print("Guitar playing")

guitar = Guitar()
guitar.play()
