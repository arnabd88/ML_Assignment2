## ----- Assignment2 : The return of the Perceptron --------------------
import sys
import re
import copy
import func
import math
import numpy


## -fold -test -margin -sanity

fold_index = -1  # default required for single training file
test_index = -1  # when testing pass  this switch
foldValue = 0    # required with fold_index
marginIndex = -1 ## default MarginList[0]; enables the CV with margins
initRandom = 0   #  enable random init. Default zero
sanityIndex = -1


LRateList = [1, 0.1, 0.01]
MarginList = [0,1,2,3,4,5]
trainFileHandle = []
sanityTable = []
testFilehandle = []

if ('-fold' in sys.argv):
	fold_index = sys.argv.index('-fold')
if('-test' in sys.argv):
	test_index = sys.argv.index('-test')
if('-sanity' in sys.argv):
	sanityIndex = sys.argv.index('-sanity')



if(sanityIndex != -1):
	sanityTable.append(open( sys.argv[sanityIndex+1], 'r+').read().splitlines())
if(fold_index != -1):
	foldValue = int(sys.argv[fold_index + 1])
	for f in range(1, int(sys.argv[fold_index + 1])+1):
		trainFileHandle.append(open( sys.argv[fold_index+1 + f], 'r+').read().splitlines())
	#print trainFileHandle
else:
	print 'Training files not provided .......... Exiting!!'
	sys.exit()

if( test_index != -1 ):
	testFilehandle.append(open( sys.argv[test_index+1], 'r+').read().splitlines())
else:
	print 'Test Data not Found..... No Testing!!'

#[XData, YData, FSize] = func.parseInfo(trainFileHandle)
#print YData, FSize



##------- Run the simple perceptron from the table-2 data(Sanity Check) --------------
def Run_Q3_3_1():
	print "//Question-3.3.1 Report (Sanity check using Table2 -----------------------------"
	[XData, YData, FSize] = func.parseInfo(sanityTable)
	maxVecLen = max(FSize)
	for i in range(1,11):
		for j in LRateList:
			[Bias, Wvec, LearningMistake] = func.SimplePerceptron(j, 0, XData, YData, maxVecLen, i, 1)
			mistakeCount = func.TestSimplePerceptron(XData, YData, Wvec, Bias, maxVecLen)
			TrainAccuracy = (float(len(XData) - mistakeCount)/len(XData))*100
			print 'Using Learning Rate = ',j,' number of epochs = ',i, 'Initialization = @ default 0'
			print '\t Learned Weight Vector    = ', Wvec
			print '\t Learned Bias             = ', Bias
			print '\t Mistakes during Learning = ', LearningMistake
			print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
	print "//End of Question-3.3.1 Report -------------------------------------------------\n\n"


##------- Run the simple perceptron on adult data for training and testing ------##
def Run_Q3_3_2_sp():
	print "//Question-3.3.2 (Simple Perceptron) Report on Adult Data -----------------------------"
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	maxVecLen = max(FSize)
	for i in range(1,5):
		for j in LRateList:
			[Bias, Wvec, LearningMistake] = func.SimplePerceptron(j, 1, XData, YData, maxVecLen, i, 1)
			mistakeCount = func.TestSimplePerceptron(XData, YData, Wvec, Bias, maxVecLen)
			TrainAccuracy = ((float(len(XData) - mistakeCount))/len(XData))*100
			print 'Using Learning Rate = ',j,' number of epochs = ',i, 'Initialization = random with mean 0 and sd 0.1'
		#	print '\t Learned Weight Vector    = ', Wvec
			print '\t Learned Bias             = ', Bias
			print '\t Mistakes during Learning = ', LearningMistake
			print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
			if(test_index != -1):
				[testXData, testYData] = func.parseInfoTest(testFilehandle, maxVecLen)
				testMistake = func.TestSimplePerceptron(testXData, testYData, Wvec, Bias, maxVecLen)
				TestAccuracy = ((float(len(testXData) - testMistake))/len(testXData))*100
				print '\t Test Accuracy            = ', TestAccuracy,'%\n'
	print "//End of Question-3.3.2 Report ------------------------------------\n\n"



##-------- Run the margin perceptron on adult data for training and testing ----##
def Run_Q3_3_2_mp():
	print "//Question-3.3.2 (Margin Perceptron) Report on Adult Data -----------------------------"
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	maxVecLen = max(FSize)
	for i in range(1,5):
		for j in LRateList:
			for k in MarginList:
				[Bias, Wvec, LearningMistake] = func.MarginPerceptron(j, 1, XData, YData, maxVecLen, i, k, 1)
				mistakeCount = func.TestMarginPerceptron(XData, YData, Wvec, Bias, maxVecLen, k)
				TrainAccuracy = ((float(len(XData) - mistakeCount))/len(XData))*100
				print 'Using Learning Rate = ',j,', number of epochs = ',i,', Margin = ', k,  ', Initialization = random with mean 0 and sd 0.1'
			#	print '\t Learned Weight Vector    = ', Wvec
				print '\t Learned Bias             = ', Bias
				print '\t Mistakes during Learning = ', LearningMistake
				print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
				if(test_index != -1):
					[testXData, testYData] = func.parseInfoTest(testFilehandle, maxVecLen)
					testMistake = func.TestMarginPerceptron(testXData, testYData, Wvec, Bias, maxVecLen, k)
					TestAccuracy = ((float(len(testXData) - testMistake))/len(testXData))*100
					print '\t Test Accuracy            = ', TestAccuracy,'%\n'
	print "//End of Question-3.3.2 Report ------------------------------------\n\n"








Run_Q3_3_1()
Run_Q3_3_2_sp()
Run_Q3_3_2_mp()

