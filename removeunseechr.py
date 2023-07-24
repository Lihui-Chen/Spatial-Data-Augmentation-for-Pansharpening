# -*- coding: utf-8 -*-
""" stripbom.py: Solve the problem about BOM """
import os
import sys
import glob

fileList = []

for dirpath, dirnames, filenames in os.walk('.'):
    for filename in filenames:
        fileList.append(os.path.join(dirpath, filename))
for tmpfile in fileList:
    if '.py' in tmpfile and '.pyc' not in tmpfile:
        with open(tmpfile, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            dirname, filename = os.path.split(tmpfile)
            file_name, extension = os.path.splitext(filename)
            new_file = os.path.join(dirname, file_name + extension)
            with open(new_file, 'w', encoding='utf-8') as ff:
                for line in lines:
                    line = line.replace(chr(65279), '')
                    ff.write(line)