import sys
import re
import copy
import math
import numpy

##---- Trims a string towards the left
def TrimLeft( trimString ):
	while( re.match(' ', trimString)):
		trimString = trimString[1:]
	return trimString


##--- concats elements of list and returns a string ---##
def concatList(l1):
	l2 = ''
	if(len(l1)!=0):
		for i in l1:
			l2 = l2+i
	return l2


##--- trims all spaces from a string
def trimStr( str1 ):
	l2 = ''
	for i in str1:
		if(i!=' '):
			l2 = l2+i
	return l2

def trimList( list1 ):
	l2 = []
	for i in list1:
		if(i != '' and i!='\n'):
			l2.append(i)
	return l2

def extendList( l1, maxSize ):
	size = maxSize - len(l1)
	newL = [0]*size
	l1.extend(newL)
	return l1


def parseInfo( rawData ):
	XData = []
	YData = []
	Fsize = []
	x_temp = []

	for flist in rawData:
		for line in flist:
			lineList = line.split()
			YData.append(int(lineList[0]))
			currMax = len(x_temp)
			x_temp = [0]*currMax
			for i in lineList[1:]:
				arg = i.split(':')
				if(int(arg[0]) > len(x_temp)):
					x_temp = extendList(x_temp, int(arg[0]))
				x_temp[int(arg[0])-1] = int(arg[1])
			XData.append(x_temp)
			Fsize.append(len(x_temp))
	return [XData, YData, Fsize]
				
		

