#!/bin/python

class Instrument:
    def play(self):
        print("Instrument playing")

class Guitar(Instrument): # Inherit from Instrument
    def play_song(self):
        print("Guitar playing a song")

guitar = Guitar()
guitar.play()
guitar.play_song()
