from keras.models import load_model
from nltk.tokenize import RegexpTokenizer
import pickle
import heapq
import numpy as np

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

#create unique char indices dict
chars=list(set(text))
indices_char = dict((c, i) for i, c in enumerate(chars))

#load the model
model = load_model('keras_next_word_model.h5')

#convert the input to a numeric vector
#SEQUENCE_LENGTH=40
#def prepare_input(text):
#    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
#    for t, char in enumerate(text):
#    return x
WORD_LENGTH=5
def prepare_input(text):
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    for t, word in enumerate(text.split()):
        x[0, t, unique_word_index[word]] = 1
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]

quotes = [
    "anything that will not be the a or whatever",
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I am not upset that you lied to me, I am upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]

for q in quotes:
    seq = " ".join(q.lower().replace(',','').strip('.').split(' ')[:5])
    print(seq)
    print(prepare_input(seq))
    print(predict_completions(seq, 5))
    print()
