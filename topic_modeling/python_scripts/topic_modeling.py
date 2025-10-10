#perform doc2vec embeddings on comments with gensim
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans

print("Loading data...<br>")

#read in the data
n_samples = 1000
data = pd.read_csv('uploads/comments.csv',header=None,names=['comment'])
comments = data['comment'][:n_samples]

keywords = list(pd.read_csv("uploads/keywords.csv"))

try:
    model = Doc2Vec.load("d2v.model")
    print("Model loaded successfully.<br>")
except FileNotFoundError:
    print("Model file not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

comment_vectors = {doc:model.infer_vector(word_tokenize(doc.lower())) for doc in comments}
comment_vectors_list = list(comment_vectors.values())

keyword_vectors = {doc:model.infer_vector(word_tokenize(doc.lower())) for doc in keywords}
keyword_vectors_list = list(keyword_vectors.values())

#determine the number of clusters with elbow method, ideally automatically
num_clusters = 5

try:
    # Attempt to create and fit the KMeans model
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(comment_vectors_list)
    print("KMeans model fitted successfully.<br><br>")
except ValueError as ve:
    print(f"ValueError: {ve}<br>")  # Handle issues with data shape or invalid parameters
except NotFittedError:
    print("Error: The model was not fitted.<br>")
except Exception as e:
    print(f"An unexpected error occurred: {e}<br>")

#print first 10 keywords
print("<b>First 10 Keywords:</b><br>", keywords[:10],"<br><br>")

#define cosine similarity
from numpy.linalg import norm
def cosine_similarity(v,w):
    return np.dot(v,w)/(norm(v)*norm(w))

#find closest bigram to centroid
centroids  = kmeans.cluster_centers_
nearest_keyword_to_centroid = []
for centroid in centroids:
    min_dist = 2**32
    nearest_keyword = ""
    for keyword, embedding in keyword_vectors.items():
        #dist = cosine_similarity(centroid, embedding)
        dist = np.linalg.norm(centroid - embedding)
        if dist < min_dist:
            min_dist = dist
            nearest_keyword = keyword
    nearest_keyword_to_centroid.append((nearest_keyword,min_dist))

print("<b>Nearest keyword to cluster centroid:</b><br>")
for nearest_keyword in nearest_keyword_to_centroid:
    print(nearest_keyword,"<br>")

print("<br>")

#label the data with the appropriate label from k-means clustering
labeled_data = list(zip(comments,kmeans.labels_))

#separate documents by label. build vocabulary for each cluster
cluster_docs = [[] for index in range(num_clusters)]
for doc, label in labeled_data:
    cluster_docs[label].append(doc)

#Use TF-IDF on each cluster to extract top two words
from sklearn.feature_extraction.text import TfidfVectorizer
def top_words_in_cluster(cluster_index, num_words=10):
    try:
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english')
        corpus_tfidf = tfidf_vectorizer.fit_transform(cluster_docs[cluster_index])
        tfidf_df = pd.DataFrame(corpus_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        tfidf_scores = tfidf_df.sum(axis=0)
        return tfidf_scores.nlargest(n=num_words)
    except IndexError:
        print(f"Error: Cluster index {cluster_index} is out of range.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


print("<b>Top TF-IDF scores on each cluster:</b>","<br>")
for index in range(num_clusters):
    print(f"<i>Cluster {index}:</i><br>")
    for word, score in top_words_in_cluster(index).items():
        print(f"({word},{score:.2f})")
    print("<br>")

print("<br>")

#find the keyword which is closest to each comment in the embedding
nearest_keyword_to_comment = []
for comment in comments[:10]:
    min_dist = 2**32
    nearest_keyword = ""
    comment_vec = comment_vectors[comment]
    for keyword, embedding in keyword_vectors.items():
        #dist = cosine_similarity(comment_vec, embedding)
        dist = np.linalg.norm(comment_vec - embedding)
        if dist < min_dist:
            min_dist = dist
            nearest_keyword = keyword
    nearest_keyword_to_comment.append((comment,nearest_keyword,min_dist))

print("<b>Nearest keyword to comment:</b><br>")
for nearest_comment in nearest_keyword_to_comment:
    print(nearest_comment,"<br>")

print("<br><br>")
