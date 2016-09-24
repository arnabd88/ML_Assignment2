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
	print '*********************************************************************************'
	print "//Question-3.3.1 Report (Sanity check using Table2 -----------------------------"
	print '*********************************************************************************'
	[XData, YData, FSize] = func.parseInfo(sanityTable)
	maxVecLen = max(FSize)
	for i in range(1,2):
		for j in LRateList:
			[Bias, Wvec, LearningMistake] = func.Perceptron(j, 0, XData, YData, maxVecLen, i, 0, 0)
			mistakeCount = func.TestPerceptron(XData, YData, Wvec, Bias, maxVecLen, 0)
			TrainAccuracy = (float(len(XData) - mistakeCount)/len(XData))*100
			print 'Using Learning Rate = ',j,' number of epochs = ',i, 'Initialization = @ default 0 , No-Shuffle'
			print '\t Learned Weight Vector    = ', Wvec
			print '\t Learned Bias             = ', Bias
			print '\t Mistakes during Learning = ', LearningMistake
			print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
	print "//End of Question-3.3.1 Report -------------------------------------------------\n\n"


##------- Run the simple perceptron on adult data for training and testing ------##
def Run_Q3_3_2_sp():
	print '*********************************************************************************'
	print "//Question-3.3.2 (Simple Perceptron-single pass) Report on Adult Data -----------------------------"
	print '*********************************************************************************'
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	maxVecLen = max(FSize)
	margin = 0
	for i in range(1,2):
		for j in LRateList:
			[Bias, Wvec, LearningMistake] = func.Perceptron(j, 1, XData, YData, maxVecLen, i, margin, 0)
			mistakeCount = func.TestPerceptron(XData, YData, Wvec, Bias, maxVecLen, margin)
			TrainAccuracy = ((float(len(XData) - mistakeCount))/len(XData))*100
			print 'Using Learning Rate = ',j,' number of epochs = Sinlge Pass,  Initialization = random with mean 0 and sd 0.1'
		#	print '\t Learned Weight Vector    = ', Wvec
			print '\t Learned Bias             = ', Bias
			print '\t Mistakes during Learning = ', LearningMistake
			print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
			if(test_index != -1):
				[testXData, testYData] = func.parseInfoTest(testFilehandle, maxVecLen)
				testMistake = func.TestPerceptron(testXData, testYData, Wvec, Bias, maxVecLen, margin)
				TestAccuracy = ((float(len(testXData) - testMistake))/len(testXData))*100
				print '\t Test Accuracy            = ', TestAccuracy,'%\n'
	print "//End of Question-3.3.2(simple Perceptron-single pass) Report ------------------------------------\n\n\n"



##-------- Run the margin perceptron on adult data for training and testing ----##
def Run_Q3_3_2_mp():
	print '*********************************************************************************'
	print "//Question-3.3.2 (Margin Perceptron-single pass) Report on Adult Data -----------------------------"
	print '*********************************************************************************'
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	maxVecLen = max(FSize)
	for i in range(1,2):
		for j in LRateList:
			for k in MarginList:
				[Bias, Wvec, LearningMistake] = func.Perceptron(j, 1, XData, YData, maxVecLen, i, k, 0)
				mistakeCount = func.TestPerceptron(XData, YData, Wvec, Bias, maxVecLen, k)
				TrainAccuracy = ((float(len(XData) - mistakeCount))/len(XData))*100
				print 'Using Learning Rate = ',j,', number of epochs = Single Pass,  Margin = ', k,  ', Initialization = random with mean 0 and sd 0.1'
			#	print '\t Learned Weight Vector    = ', Wvec
				print '\t Learned Bias             = ', Bias
				print '\t Mistakes during Learning = ', LearningMistake
				print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
				if(test_index != -1):
					[testXData, testYData] = func.parseInfoTest(testFilehandle, maxVecLen)
					testMistake = func.TestPerceptron(testXData, testYData, Wvec, Bias, maxVecLen, k)
					TestAccuracy = ((float(len(testXData) - testMistake))/len(testXData))*100
					print '\t Test Accuracy            = ', TestAccuracy,'%\n'
	print "//End of Question-3.3.2(Margin Perceptron-single pass) Report ------------------------------------\n\n\n"



##-------- Run the simple perceptron on adult data in Batch Mode for 3 to 5 epochs with no shuffle
def Run_Q3_3_3_sp_batch_noShuffle():
	print '***************************************************************************************************'
	print "//Question-3.3.2 (Simple Perceptron-batch 3,4,5) No Shuffle Report on Adult Data -----------------------------"
	print '***************************************************************************************************'
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	maxVecLen = max(FSize)
	margin = 0
	for i in range(3,6):
		for j in LRateList:
			[Bias, Wvec, LearningMistake] = func.Perceptron(j, 1, XData, YData, maxVecLen, i, margin, 0)
			mistakeCount = func.TestPerceptron(XData, YData, Wvec, Bias, maxVecLen, margin)
			TrainAccuracy = ((float(len(XData) - mistakeCount))/len(XData))*100
			print 'Using Learning Rate = ',j,' number of epochs = ',i,'  Initialization = random with mean 0 and sd 0.1'
		#	print '\t Learned Weight Vector    = ', Wvec
			print '\t Learned Bias             = ', Bias
			print '\t Mistakes during Learning = ', LearningMistake
			print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
			if(test_index != -1):
				[testXData, testYData] = func.parseInfoTest(testFilehandle, maxVecLen)
				testMistake = func.TestPerceptron(testXData, testYData, Wvec, Bias, maxVecLen, margin)
				TestAccuracy = ((float(len(testXData) - testMistake))/len(testXData))*100
				print '\t Test Accuracy            = ', TestAccuracy,'%\n'
	print "//End of Question-3.3.2(simple Perceptron-Batch Mode-NoShuffle) Report ------------------------------------\n\n\n"




##-------- Run the margin perceptron on adult data in Batch Mode for 3 to 5 epochs with no shuffle ----##
def Run_Q3_3_3_mp_batch_noShuffle():
	print '***************************************************************************************************'
	print "//Question-3.3.2 (Margin Perceptron-batch 3,4,5) No Shuffle Report on Adult Data -----------------------------"
	print '***************************************************************************************************'
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	maxVecLen = max(FSize)
	for i in range(3,6):
		for j in LRateList:
			for k in MarginList:
				[Bias, Wvec, LearningMistake] = func.Perceptron(j, 1, XData, YData, maxVecLen, i, k, 0)
				mistakeCount = func.TestPerceptron(XData, YData, Wvec, Bias, maxVecLen, k)
				TrainAccuracy = ((float(len(XData) - mistakeCount))/len(XData))*100
				print 'Using Learning Rate = ',j,', number of epochs = ',i,'  Margin = ', k,  ', Initialization = random with mean 0 and sd 0.1'
			#	print '\t Learned Weight Vector    = ', Wvec
				print '\t Learned Bias             = ', Bias
				print '\t Mistakes during Learning = ', LearningMistake
				print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
				if(test_index != -1):
					[testXData, testYData] = func.parseInfoTest(testFilehandle, maxVecLen)
					testMistake = func.TestPerceptron(testXData, testYData, Wvec, Bias, maxVecLen, k)
					TestAccuracy = ((float(len(testXData) - testMistake))/len(testXData))*100
					print '\t Test Accuracy            = ', TestAccuracy,'%\n'
	print "//End of Question-3.3.2(Margin Perceptron-Batch Mode-NoShuffle) Report ------------------------------------\n\n\n"





##-------- Run the simple perceptron on adult data in Batch Mode for 3 to 5 epochs with shuffle
def Run_Q3_3_3_sp_batch_Shuffle():
	print '***************************************************************************************************'
	print "//Question-3.3.2 (Simple Perceptron-batch 3,4,5) With Shuffle Report on Adult Data -----------------------------"
	print '***************************************************************************************************'
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	maxVecLen = max(FSize)
	margin = 0
	for i in range(3,6):
		for j in LRateList:
			[Bias, Wvec, LearningMistake] = func.Perceptron(j, 1, XData, YData, maxVecLen, i, margin, 1)
			mistakeCount = func.TestPerceptron(XData, YData, Wvec, Bias, maxVecLen, margin)
			TrainAccuracy = ((float(len(XData) - mistakeCount))/len(XData))*100
			print 'Using Learning Rate = ',j,' number of epochs = ',i,'  Initialization = random with mean 0 and sd 0.1, With Shuffle'
		#	print '\t Learned Weight Vector    = ', Wvec
			print '\t Learned Bias             = ', Bias
			print '\t Mistakes during Learning = ', LearningMistake
			print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
			if(test_index != -1):
				[testXData, testYData] = func.parseInfoTest(testFilehandle, maxVecLen)
				testMistake = func.TestPerceptron(testXData, testYData, Wvec, Bias, maxVecLen, margin)
				TestAccuracy = ((float(len(testXData) - testMistake))/len(testXData))*100
				print '\t Test Accuracy            = ', TestAccuracy,'%\n'
	print "//End of Question-3.3.2(simple Perceptron-Batch Mode-Shuffle) Report ------------------------------------\n\n\n"




##-------- Run the margin perceptron on adult data in Batch Mode for 3 to 5 epochs with no shuffle ----##
def Run_Q3_3_3_mp_batch_Shuffle():
	print '***************************************************************************************************'
	print "//Question-3.3.2 (Margin Perceptron-batch 3,4,5) With Shuffle Report on Adult Data -----------------------------"
	print '***************************************************************************************************'
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	maxVecLen = max(FSize)
	for i in range(3,6):
		for j in LRateList:
			for k in MarginList:
				[Bias, Wvec, LearningMistake] = func.Perceptron(j, 1, XData, YData, maxVecLen, i, k, 0)
				mistakeCount = func.TestPerceptron(XData, YData, Wvec, Bias, maxVecLen, k)
				TrainAccuracy = ((float(len(XData) - mistakeCount))/len(XData))*100
				print 'Using Learning Rate = ',j,', number of epochs = ',i,'  Margin = ', k,  ', Initialization = random with mean 0 and sd 0.1, with shuffle'
			#	print '\t Learned Weight Vector    = ', Wvec
				print '\t Learned Bias             = ', Bias
				print '\t Mistakes during Learning = ', LearningMistake
				print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
				if(test_index != -1):
					[testXData, testYData] = func.parseInfoTest(testFilehandle, maxVecLen)
					testMistake = func.TestPerceptron(testXData, testYData, Wvec, Bias, maxVecLen, k)
					TestAccuracy = ((float(len(testXData) - testMistake))/len(testXData))*100
					print '\t Test Accuracy            = ', TestAccuracy,'%\n'
	print "//End of Question-3.3.2(Margin Perceptron-Batch Mode-Shuffle) Report ------------------------------------\n\n\n"







##-------- Run the aggressive perceptron on adult data set for training and testing ----##
def Run_Q3_grad_with_noShuffle():
	print '*********************************************************************************'
	print "//Question-3: For Grads (Aggressive Perceptron-batch 3,4,5) With No Shuffle Report on Adult Data Set -------------"
	print '*********************************************************************************'
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	maxVecLen = max(FSize)
	for i in range(3,6):
		for k in MarginList:
			[Bias, Wvec, LearningMistake] = func.AggressivePerceptron(1, XData, YData, maxVecLen, i, k, 0)
			mistakeCount = func.TestMarginPerceptron(XData, YData, Wvec, Bias, maxVecLen, k)
			TrainAccuracy = ((float(len(XData) - mistakeCount))/len(XData))*100
			print 'Using Dynamic Learning rate, number of epochs = ',i,', Margin = ', k,  ', Initialization = random with mean 0 and sd 0.1, no Shuffle'
			#print '\t Learned Weight Vector    = ', Wvec
			print '\t Learned Bias             = ', Bias
			print '\t Mistakes during Learning = ', LearningMistake
			print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
			if(test_index != -1):
				[testXData, testYData] = func.parseInfoTest(testFilehandle, maxVecLen)
				testMistake = func.TestMarginPerceptron(testXData, testYData, Wvec, Bias, maxVecLen, k)
				TestAccuracy = ((float(len(testXData) - testMistake))/len(testXData))*100
				print '\t Test Accuracy            = ', TestAccuracy,'%\n'

	print "//End of Question-3: For Grads(Aggressive Perceptron-Batch Mode-NoShuffle Report -----------------------------\n\n\n"




def Run_Q3_grad_with_Shuffle():
	print '*********************************************************************************'
	print "//Question-3: For Grads (Aggressive Perceptron-batch 3,4,5) With Shuffle Report on Adult Data Set -------------"
	print '*********************************************************************************'
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	maxVecLen = max(FSize)
	for i in range(3,6):
		for k in MarginList:
			[Bias, Wvec, LearningMistake] = func.AggressivePerceptron(1, XData, YData, maxVecLen, i, k, 1)
			mistakeCount = func.TestMarginPerceptron(XData, YData, Wvec, Bias, maxVecLen, k)
			TrainAccuracy = ((float(len(XData) - mistakeCount))/len(XData))*100
			print 'Using Dynamic Learning rate, number of epochs = ',i,', Margin = ', k,  ', Initialization = random with mean 0 and sd 0.1, with Shuffle'
			#print '\t Learned Weight Vector    = ', Wvec
			print '\t Learned Bias             = ', Bias
			print '\t Mistakes during Learning = ', LearningMistake
			print '\t Learning Accuracy        = ', TrainAccuracy,'%\n'
			if(test_index != -1):
				[testXData, testYData] = func.parseInfoTest(testFilehandle, maxVecLen)
				testMistake = func.TestMarginPerceptron(testXData, testYData, Wvec, Bias, maxVecLen, k)
				TestAccuracy = ((float(len(testXData) - testMistake))/len(testXData))*100
				print '\t Test Accuracy            = ', TestAccuracy,'%\n'

	print "//End of Question-3: For Grads(Aggressive Perceptron-Batch Mode-Shuffle Report -----------------------------\n\n\n"





if ('-sanity' in sys.argv):
	Run_Q3_3_1()
if('-q32' in sys.argv):
	Run_Q3_3_2_sp()
	Run_Q3_3_2_mp()
if('-q33' in sys.argv):
	Run_Q3_3_3_sp_batch_noShuffle()
	Run_Q3_3_3_mp_batch_noShuffle()
	Run_Q3_3_3_sp_batch_Shuffle()
	Run_Q3_3_3_mp_batch_Shuffle()
if('-q3grad' in sys.argv):
	Run_Q3_grad_with_noShuffle()
	Run_Q3_grad_with_Shuffle()
#Run_Q3_grad()

