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
print(prev_words[0])
print(next_words[0])
