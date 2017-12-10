#!/bin/python

import sys

from PyQt4 import QtSql

from PyQt4.QtGui import *
from PyQt4.QtSql import *

class Pass:
    def __init__(self):
        self.db = QtSql.QSqlDatabase.addDatabase('QMYSQL')
        self.db.setHostName("localhost")
        self.db.setDatabaseName("size_db")
        self.db.setUserName("alejandro")

        self.win = QDialog()
        self.win.setWindowTitle("Database connection")
        self.win.setGeometry(100,100,200,100)

        self.label = QLabel(self.win)
        self.label.setText("Password")
        self.label.move(5,25)

        self.e1 = QLineEdit(self.win)
        self.e1.setEchoMode(QLineEdit.Password)
        self.e1.move(70,20)

        self.b1 = QPushButton(self.win)
        self.b1.setText("Connect")
        self.b1.move(70,50)
        self.b1.clicked.connect(self.b1_clicked)

        self.win.show()

    def b1_clicked(self):
        self.db.setPassword(self.e1.text())

        if self.db.open():
            print("Connection established")
            self.win2 = QDialog()
            self.win2.setWindowTitle("Bill")
            self.win2.setGeometry(300,300,200,100)

            self.win.close()
            self.win2.show()

            self.db.close()
        else:
            self.win3 = QDialog()
            self.win3.setWindowTitle("Error")
            self.win3.setGeometry(200,200,200,100)

            self.label = QLabel(self.win3)
            self.label.setText("Bad Password!")
            self.label.move(65,30)

            self.b2 = QPushButton(self.win3)
            self.b2.setText("Ok")
            self.b2.move(70,50)
            self.b2.clicked.connect(self.win3.close)

            self.win3.show()

        self.e1.clear()

#       query = QtSql.QSqlQuery()
#       val = query.exec('insert into elements (id_bill,descr,quant,cost) values (1,"something",2,30);')
#       print(val)

app = QApplication(sys.argv)
passw = Pass()
sys.exit(app.exec_())
