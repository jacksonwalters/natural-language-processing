import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk

df_sms = pd.read_csv('spam.csv',encoding='latin-1')

df_sms = df_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df_sms = df_sms.rename(columns={"v1":"label", "v2":"sms"})

#create a new column called length which is the length of the sms
df_sms['length'] = df_sms['sms'].apply(len)

#plot the distribution of lengths of texts
import matplotlib.pyplot as plt
import seaborn as sns
df_sms['length'].plot(bins=50, kind='hist')
#plt.show()

#plot spam vs. ham length distributions to see if there's a difference
#spam texts are longer on average
df_sms.hist(column='length', by='label', bins=50,figsize=(10,4))
#plt.show()

#label "ham" by 0, spam by 1
df_sms.loc[:,'label'] = df_sms.label.map({'ham':0, 'spam':1})

#use countvectorizer to get word counts
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()

documents = df_sms['sms'].values
count_vector.fit(documents)
count_vector.get_feature_names_out()

doc_array = count_vector.transform(documents).toarray()

frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names_out())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_sms['sms'],
                                                    df_sms['label'],test_size=0.20,
                                                    random_state=1)

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix.
testing_data = count_vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)

#evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: {}'.format(accuracy_score(y_test, predictions)))
print('Precision score: {}'.format(precision_score(y_test, predictions)))
print('Recall score: {}'.format(recall_score(y_test, predictions)))
print('F1 score: {}'.format(f1_score(y_test, predictions)))
