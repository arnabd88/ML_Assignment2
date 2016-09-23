## ----- Assignment2 : The return of the Perceptron --------------------
import sys
import re
import copy
import func
import math
import numpy


## -fold -test -margin 

fold_index = -1  # default required for single training file
test_index = -1  # when testing pass  this switch
foldValue = 0    # required with fold_index
marginIndex = -1 ## default MarginList[0]; enables the CV with margins
initRandom = 0   #  enable random init. Default zero


LRateList = [1, 0.1, 0.01]
MarginList = [0,1,2,3,4,5]
trainFileHandle = []

if ('-fold' in sys.argv):
	fold_index = sys.argv.index('-fold')
if('-test' in sys.argv):
	test_index = sys.argv.index('-test')


if(fold_index != -1):
	foldValue = int(sys.argv[fold_index + 1])
	for f in range(1, int(sys.argv[fold_index + 1])+1):
		trainFileHandle.append(open( sys.argv[fold_index+1 + f], 'r+').read().splitlines())
	#print trainFileHandle
else:
	print 'Training files not provided .......... Exiting!!'
	sys.exit()

[XData, YData, FSize] = func.parseInfo(trainFileHandle)
print max(FSize)




