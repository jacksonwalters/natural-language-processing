# text-binary-classifier
Practice training neural networks for text classification with Tensorflow/Keras.

Includes MNIST benchmark example. Jupyter notebooks implement multiple models to perform binary classification on sentences. 

Begins with a simple logistic regression, introduces text vectorization, then adds dense layers using Keras, pre-trained text embeddings (trainable and not), a 1d CNN layer, and finally an RNN layer (LSTM).

Data are 6,000 labeled tweets each from Joe Biden and Donald Trump, and reviews from IMDb, Yelp, and Amazon labeled by sentiment.

Adapted from https://realpython.com/python-keras-text-classification/ and https://www.tensorflow.org/guide/keras/rnn.
