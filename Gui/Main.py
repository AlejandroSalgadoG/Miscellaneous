#!/bin/python

import sys

from PyQt4.QtGui import *
from Creator import *
from Graphic import *

app = QApplication(sys.argv)

window = WindowCreator().create()
new_button = NewButtonCreator().create(window)

window.add(new_button)

window.draw()

sys.exit(app.exec_())
