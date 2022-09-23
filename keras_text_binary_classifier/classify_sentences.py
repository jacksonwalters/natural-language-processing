from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

#PLACEHOLDER. use Keras text classifier
#https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/text_classification_with_hub.ipynb
#would need to upload formatted data for political bias classification
if __name__ == "__main__":
    #load labeled data
    filepath = '/Users/jacksonwalters/tensorflow_datasets/labeled_tweets/all_labeled_tweets.txt'
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    sentences = df['sentence'].values
    y = df['label'].values  #tweet sentence sentiment labels. 0 = negative, 1 = positive
    #split the sentences into training data and test data
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
    #build vectorizer and set padding param
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 100

    #load the model and test on examples
    model = keras.models.load_model('cnn_model') #load the trained CNN text classifier
    examples = ["fake news","mike pence","witch hunt","law and order","health care","kamala harris"]
    X_ex_sent = tokenizer.texts_to_sequences(examples)
    X_ex_sent = tokenizer.texts_to_sequences(examples)
    X_ex_sent = pad_sequences(X_ex_sent, padding='post', maxlen=maxlen)
    predictions = model.predict(X_ex_sent)
    pred_dict = {examples[i]:predictions[i][0] for i in range(len(examples))}
    print(pred_dict)
