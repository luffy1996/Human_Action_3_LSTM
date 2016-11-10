from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

###############################################

def readdata(path='/home/luffy/Desktop/assignment_3/testing/nietzsche.txt'):
	#path = '/home/luffy/Desktop/assignment_3/testing/nietzsche.txt'
	text = open(path).read().lower()
	print('corpus length:', len(text))
	print('#'*50)
	print('Data Reading Complete')
	print('#'*50)
	return text 

def lstm(path='/home/luffy/Desktop/assignment_3/testing/nietzsche.txt',nb_epoch=20):
	text = readdata();
	

if __name__ == '__main__':
	lstm(nb_epoch=5)