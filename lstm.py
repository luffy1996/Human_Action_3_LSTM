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

def lstm(path='/home/luffy/Desktop/assignment_3/testing/nietzsche.txt',nb_epoch=20, maxlen=50,step = 5):
	text = readdata();
	chars = sorted(list(set(text)))
	''' Here I am reading all types of characters that has been used in the text file. '''
	''' To find the number of characters in the text file uncomment the following command'''
	#print('total chars:', len(chars))
	
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars)) 

	''' Maxlenght and step has been provided in the argument of funtion defination.  '''
	sentences = []
	next_character = []

	for i in range(0, len(text) - maxlen, step):
		'''You need to create a sub array of senstence with maximum length as specified'''
		sentences.append(text[i: i + maxlen])
		'''now you need to store the next character that will appear in the text file'''
		next_character.append(text[i + maxlen])
	print('nb sequences:', len(sentences))
	print('#'*50)
	print ('Vectorization of data')
	print('#'*50)
	X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
	y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
	'''Intializing the boolean weights to 0'''
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			X[i, t, char_indices[char]] = 1
		y[i, char_indices[next_character[i]]] = 1

	#print('#'*50)
	print ('Building Model')
	print('#'*50)	
	


if __name__ == '__main__':
	lstm(nb_epoch=5)