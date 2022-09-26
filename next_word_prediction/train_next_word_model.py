import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

path = '1661-0.txt'
text = open(path).read().lower()
print('corpus length:', len(text))

#clean the words using a regex
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

#create a dict mapping unique words to their index
#presumably their first occurence
unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

#create a window of WORD_LENGTH previous words
WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])

#create feature vectors X, Y
#X is keeping track of combinations of WORD_LENGTH words
#Y is keeping track of when a WORD_LENGTH combination of words yields the next word
X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1

#let's use a LSTM, long short term memory. This is an RNN which stores states
#and uses a feedback loop
#build the model
model = Sequential()
model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))

#train the model
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

#save model for future use
model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))
