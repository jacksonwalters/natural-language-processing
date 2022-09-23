#DATA: 6,000 labled tweets before 2020 election from Biden=0, Trump=1.
#VECTORIZER: tokenizer. vocab size 19,415.
#MODEL: 1-d CNN with a global max-pool layer.

if __name__ == "__main__":

    #load the data
    import os
    import pandas as pd

    os.chdir('/Users/jacksonwalters/tensorflow_datasets')
    filepath_dict = {'tweet':   'labeled_tweets/all_labeled_tweets.txt'}

    df_list = []
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
        df['source'] = source  # Add another column filled with the source name
        df_list.append(df)

        df = pd.concat(df_list)
    print(df.iloc[0])

    #vectorize the data and split into train/test
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split

    tokenizer = Tokenizer(num_words=5000)
    #slice the df to get yelp sentences
    df_tweet = df[df['source'] == 'tweet']
    sentences = df_tweet['sentence'].values
    #tweet sentence sentiment labels. 0 = negative, 1 = positive
    y = df_tweet['label'].values
    #split the sentences into training data and test data
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
    #vectorize the sentences
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)
    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1
    #pad_sequences
    maxlen = 100
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    print(vocab_size)

    #plotting function
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    def plot_history(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

    #construct the model
    from keras import Sequential
    from keras import layers

    embedding_dim = 100

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()

    #train the model
    history = model.fit(X_train, y_train,
                    epochs=2,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history)

    #test examples
    ex_sent = "fake news"
    X_ex_sent = tokenizer.texts_to_sequences([ex_sent])
    X_ex_sent = pad_sequences(X_ex_sent, padding='post', maxlen=maxlen)
    print(model.predict(X_ex_sent))

    #save the model
    model.save('/Users/jacksonwalters/Desktop/cnn_model')
