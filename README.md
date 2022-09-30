# nlp
NLP practice projects and examples. All data files in .gitignore.

Example projects: https://medium.com/coders-camp/20-machine-learning-projects-on-nlp-582effe73b9c

ML on M1: https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706

For Keras/TensorFlow on an Apple M1, I'm using conda miniforge with TensorFlow installed in a virtual environment (conda activate mlp, after install). Much of the NLP uses NLTK rather than spaCy, with separately downloaded Glove word embeddings.


1. Binary *sentence classification* via CNN/RNN on Biden/Trump tweets.
2. *Text summarization* via Glove embeddings, cosine similarity matrix, and pagerank.
3. *Sentiment analysis* via logistic regression.
4. *Keyword extraction* via TF-IDF.
5. *Hate speech detection* using SGDClassifier.
6. *Next word prediction* using Keras/TF.
7. *Spam detection* using SMS data, CountVectorizer (bag-of-words), and Multinomial Naive Bayes. 
8. *Text classification, part 2* using TF/Keras, IMDB reviews for positive/negative sentiment.
9. *Spelling correction* using TextBlob (quick).
10. *Named entity recognition* via LSTM. Too long to train on M1.
