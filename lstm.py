from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

from keras.models import load_model
from keras.models import model_from_json
###############################################

def readdata(path='/home/luffy/Desktop/assignment_3/human_action_3.txt'):
	#path = '/home/luffy/Desktop/assignment_3/human_action_3.txt'
	text = open(path).read().lower()
	print('corpus length:', len(text))
	print('#'*50)
	print('Data Reading Complete')
	print('#'*50)
	return text 

###############################################
text = readdata();
chars = sorted(list(set(text)))
''' Here I am reading all types of characters that has been used in the text file. '''
''' To find the number of characters in the text file uncomment the following command'''
print('total chars:', len(chars))
	
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars)) 

''' Maxlenght and step has been provided in the argument of funtion defination.  '''
sentences = []
next_character = []
maxlen = 50
step =3

for i in range(0, len(text) - maxlen, step):
	'''You need to create a sub array of senstence with maximum length as specified'''
	sentences.append(text[i: i + maxlen])
	'''now you need to store the next character that will appear in the text file'''
	next_character.append(text[i + maxlen])
print('nb sequences:', len(sentences))
print ('Vectorization of data')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
'''Intializing the boolean weights to 0'''
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		X[i, t, char_indices[char]] = 1
	y[i, char_indices[next_character[i]]] = 1



def lstm(path='/home/luffy/Desktop/assignment_3/human_action_3.txt',nb_epoch=20, batch_size=400):
	
	#print('#'*50)
	print ('Building Model')
	model=Sequential()
	model.add(LSTM(128, input_shape=(maxlen, len(chars))))
	model.add(Dense(len(chars)))
	model.add(Activation('softmax'))
	model.load_weights("lstm_model_for_human_action_book.h5")
	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
	model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch)
	''' Sving the model in .json format and weights in HDF5 format'''
	model_json = model.to_json()
	with open("lstm_model_for_human_action_book.json", "w") as json_file:
		json_file.write(model_json)
		# serialize weights to HDF5
	model.save_weights("lstm_model_for_human_action_book.h5")
	print("Saved model to disk")
	print('#'*50)

def generate_text():
	json_file = open('lstm_model_for_human_action_book.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights("lstm_model_for_human_action_book.h5")
	print("Loaded model from disk")
	#########################################
	starting_index = random.randint(0, len(text) - maxlen - 1)

	print()
	generated = ''
	sentence = text[starting_index: starting_index + maxlen]
	generated += sentence
	print('----- Generating a sentence :  "' + sentence + '"')
	sys.stdout.write(generated)
	diversity = 0.2
	for i in range(400):
		x = np.zeros((1, maxlen, len(chars)))
		for t, char in enumerate(sentence):
			x[0, t, char_indices[char]] = 1.

		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, diversity)
		next_char = indices_char[next_index]

		generated += next_char
		sentence = sentence[1:] + next_char

		sys.stdout.write(next_char)
		sys.stdout.flush()
	print()

def sample(preds,temperature = 1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


if __name__ == '__main__':
	'''In case you want to explicitely train your data uncomment the command lstm'''
	lstm(nb_epoch=15,batch_size=3000)
	generate_text()