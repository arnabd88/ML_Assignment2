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


def sgn( value, margin ):
	if( value >= margin):
		return 1
	else:
		return -1

def extendList( l1, maxSize ):
	size = maxSize - len(l1)
	newL = [0]*size
	l1.extend(newL)
	return l1


def permuteDataLabel( xdata, ylabel ):
	newdata = []
	newlabel = []
	shufIdx = numpy.random.permutation(len(xdata))
	if( len(xdata) != len(ylabel)):
		print 'Error......... Mismatch in data and label size..... Exiting'
		sys.exit()

	for i in shufIdx:
		newdata.append(xdata[i])
		newlabel.append(ylabel[i])

	return [newdata, newlabel]
	


def parseInfo( rawData ):
	XData = []
	YData = []
	Fsize = []
	x_temp = []
	seenwt = []

	for flist in rawData:
		for line in flist:
			lineList = line.split()
			YData.append(int(lineList[0]))
			currMax = len(x_temp)
			x_temp = [0]*currMax
			for i in lineList[1:]:
				arg = i.split(':')
				seenwt.append(int(arg[0]))
				if(int(arg[0])+1 > len(x_temp)):
					x_temp = extendList(x_temp, int(arg[0])+1)
				x_temp[int(arg[0])] = int(arg[1])
			XData.append(x_temp)
			Fsize.append(len(x_temp))
	return [XData, YData, Fsize, seenwt]

def parseInfoTest( testData, maxSize ):
	XData  = []
	YData  = []
	Fsize  = []
	x_temp = []

	for flist in testData:
		for line in flist:
			lineList = line.split()
			YData.append(int(lineList[0]))
			x_temp = [0]*maxSize
			for i in lineList[1:]:
				arg = i.split(':')
				if(int(arg[0]) < maxSize):
					x_temp[int(arg[0])] = int(arg[1])
			XData.append(x_temp)
	return [XData, YData]


def AggressivePerceptron( initEn, xdata, ydata, wsize, epochs, margin, enShuffle, seenwt):
	wvec = []
	bias = 0
	mistakeCounter = 0
	eta = 0
	##---initialize---
	if(initEn==0):
		wvec = [0]*wsize
		bias = 0
	else:
		##--- using a normal distribution of mean 0 and sd 0.1
		wvec = numpy.random.normal(0,0.01,wsize+1)
		bias = wvec[0]
		wvec = wvec[1:]
		for i in range(0,len(wvec)):
			if(i in seenwt):
				wvec[i] = 0
	##-- Make the prediction (Evaluate wx+b) --##
		for i in range(0,epochs):
			wtxSum = bias
			looprange = len(wvec)
			if(enShuffle==1):
				[xdata, ydata] = permuteDataLabel(xdata, ydata)
			for i in range(0,len(xdata)): ##looping over each of the examples
				xvec = xdata[i]
				ylabel = ydata[i]
				wtxSum = bias
				if(len(xvec) < len(wvec)):
					looprange = len(xvec)
				for i in range(0,looprange):
					wtxSum = wtxSum + xvec[i]*wvec[i]
				if(ylabel*wtxSum <= margin):
				#if(sgn(wtxSum, margin)*ylabel < 0): ##Incorrect prediction
					mistakeCounter = mistakeCounter + 1
					##--- Part of aggressive update to bias and weights ---
					xtx = numpy.dot( numpy.array(xvec),  numpy.array(xvec).transpose())
					eta = float( margin - ylabel*wtxSum)/(xtx + 1)
					#print 'Margin = ', margin
					#print 'yalebl = ', ylabel
					#print 'wtxSum = ', wtxSum
					#print 'xvec = ',xvec
					#print 'XTX = ', xtx
					#print "Current Eta = ", eta, '\n\n'
					if(len(xvec) < len(wvec)):
						xvec = extendList(xvec, len(wvec))
					correction = map(lambda x: x*ylabel*eta, xvec)
					wvec = map(sum, zip(wvec, correction))
					bias = bias + eta*ylabel

		return [bias, wvec, mistakeCounter]




def Perceptron(lr, initEn, xdata, ydata, wsize, epochs, margin, enShuffle, seenwt ):
	wvec = []
	bias = 0
	mistakeCounter = 0
	##---initialize---
	if(initEn==0):
		wvec = [0]*wsize
		bias = 0
	else:
		##--- using a normal distribution of mean 0 and sd 0.1
		wvec = numpy.random.normal(0, 0.01, wsize+1)
		bias = wvec[0]
		wvec = wvec[1:]
		for i in range(0,len(wvec)):
			if(i in seenwt):
				wvec[i] = 0
	##-- Make the prediction (Evaluate wx+b) -- ##
	for i in range(0,epochs):
		wtxSum = bias
		looprange = len(wvec)
		if(enShuffle == 1):
			[xdata, ydata] = permuteDataLabel(xdata, ydata)
		for i in range(0,len(xdata)):  ## looping over each of the examples
			xvec = xdata[i]
			ylabel = ydata[i]
			wtxSum = bias
			if(len(xvec) < len(wvec)):
				looprange = len(xvec)
			for i in range(0,looprange):
				wtxSum = wtxSum + xvec[i]*wvec[i]
			if(ylabel*wtxSum <= margin):
			#if(sgn(wtxSum, margin)*ylabel < 0): ## Incorrect prediction 
				mistakeCounter = mistakeCounter + 1
				## update the bias and weights
				if(len(xvec) < len(wvec)):
					xvec = extendList(xvec,len(wvec))
				correction = map(lambda x: x*ylabel*lr, xvec)
				wvec = map(sum, zip(wvec, correction))
				bias = bias + lr*ylabel
	return [bias, wvec, mistakeCounter]




def MarginPerceptron(lr, initEn, xdata, ydata, wsize, epochs, margin, enShuffle ):
	wvec = []
	bias = 0
	mistakeCounter = 0
	##---initialize---
	if(initEn==0):
		wvec = [0]*wsize
		bias = 0
	else:
		##--- using a normal distribution of mean 0 and sd 0.1
		wvec = numpy.random.normal(0, 0.01, wsize+1)
		bias = wvec[0]
		wvec = wvec[1:]
	##-- Make the prediction (Evaluate wx+b) -- ##
	for i in range(0,epochs):
		wtxSum = bias
		looprange = len(wvec)
		if(enShuffle == 1):
			[xdata, ydata] = permuteDataLabel(xdata, ydata)
		for i in range(0,len(xdata)):  ## looping over each of the examples
			xvec = xdata[i]
			ylabel = ydata[i]
			wtxSum = bias
			if(len(xvec) < len(wvec)):
				looprange = len(xvec)
			for i in range(0,looprange):
				wtxSum = wtxSum + xvec[i]*wvec[i]
			if(sgn(wtxSum, margin)*ylabel < 0): ## Incorrect prediction 
				mistakeCounter = mistakeCounter + 1
				## update the bias and weights
				if(len(xvec) < len(wvec)):
					xvec = extendList(xvec,len(wvec))
				correction = map(lambda x: x*ylabel*lr, xvec)
				wvec = map(sum, zip(wvec, correction))
				bias = bias + lr*ylabel
	return [bias, wvec, mistakeCounter]


## lr      = learning rate, initEn = randomInitialization Enabled
## xdata   = set of training examples [[],[],...]
## ydata   = set of labels of the training examples [,,,...]
## setBias = may be added, not decided
## wsize   = size of the weight vector
## epochs  = number of epochs  required

def SimplePerceptron(lr, initEn, xdata, ydata, wsize, epochs, enShuffle ):
	wvec = []
	bias = 0
	mistakeCounter = 0
	##---initialise ---
	if(initEn == 0): ## set bias and weight to zero
		wvec = [0]*wsize
		bias = 0
	else:
		##--- using a normal distribution of mean 0 and sd 0.1
		wvec = numpy.random.normal(0, 0.01, wsize+1)
		bias = wvec[0]
		wvec = wvec[1:]
	##-- Make the prediction (Evaluate wx+b) -- ##
	for i in range(0,epochs):
		wtxSum = bias
		looprange = len(wvec)
		if(enShuffle == 1):
			[xdata, ydata] = permuteDataLabel(xdata, ydata)
		for i in range(0,len(xdata)):  ## looping over each of the examples
			xvec = xdata[i]
			ylabel = ydata[i]
			wtxSum = bias
			if(len(xvec) < len(wvec)):
				looprange = len(xvec)
			for i in range(0,looprange):
				wtxSum = wtxSum + xvec[i]*wvec[i]
			if(sgn(wtxSum, 0)*ylabel < 0): ## Incorrect prediction 
				mistakeCounter = mistakeCounter + 1
				## update the bias and weights
				if(len(xvec) < len(wvec)):
					xvec = extendList(xvec,len(wvec))
				correction = map(lambda x: x*ylabel*lr, xvec)
				wvec = map(sum, zip(wvec, correction))
				bias = bias + lr*ylabel
	return [bias, wvec, mistakeCounter]

## Takes test data and makes predictions based on the 


def TestPerceptron( xdata, ydata, wvec, bias, wsize, margin):
	mistakeCounter = 0
	wtxSum = bias
	looprange = len(wvec)
	for i in range(0,len(xdata)):
		xvec   = xdata[i]
		ylabel = ydata[i]
		wtxSum = bias
		if(len(xvec) < len(wvec)):
			looprange = len(xvec)
		for i in range(0,looprange):
			wtxSum = wtxSum + xvec[i]*wvec[i]
		if(ylabel*wtxSum <= margin):
		#if(sgn(wtxSum, margin)*ylabel < 0):
			mistakeCounter = mistakeCounter + 1
	return mistakeCounter


def TestSimplePerceptron(xdata, ydata, wvec, bias, wsize):
	mistakeCounter = 0
	wtxSum = bias
	looprange = len(wvec)
	for i in range(0,len(xdata)):
		xvec   = xdata[i]
		ylabel = ydata[i]
		wtxSum = bias
		if(len(xvec) < len(wvec)):
			looprange = len(xvec)
		for i in range(0,looprange):
			wtxSum = wtxSum + xvec[i]*wvec[i]
		if(sgn(wtxSum, 0)*ylabel < 0):
			mistakeCounter = mistakeCounter + 1
	return mistakeCounter
	

def TestMarginPerceptron(xdata, ydata, wvec, bias, wsize, margin):
	mistakeCounter = 0
	wtxSum = bias
	looprange = len(wvec)
	for i in range(0,len(xdata)):
		xvec   = xdata[i]
		ylabel = ydata[i]
		wtxSum = bias
		if(len(xvec) < len(wvec)):
			looprange = len(xvec)
		for i in range(0,looprange):
			wtxSum = wtxSum + xvec[i]*wvec[i]
		if(sgn(wtxSum, margin)*ylabel < 0):
			mistakeCounter = mistakeCounter + 1
	return mistakeCounter
	
	

		

