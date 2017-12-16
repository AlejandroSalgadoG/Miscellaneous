#!/bin/python

import xlrd
import os
import sys

data = sys.argv[1]

data_file = xlrd.open_workbook(data)
sheet = data_file.sheet_by_index(0)
row = sheet.row(4)

for idx, cell_obj in enumerate(row):
    print('(%s) %s' % (idx, cell_obj.value))

