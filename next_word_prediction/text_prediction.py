import matplotlib.pyplot as plt
import pickle

model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb"))

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
