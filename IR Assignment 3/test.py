import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
from gensim.models import Word2Vec
from collections import Counter
import re
import string
import math

path = './Doc50'
documents = []
labels = []

# Store all the files in the documents array
def load_documents():
    files = os.listdir(path)

    for name in files:
        direc = os.path.join(path, name)
        with open(direc, 'r') as file:
            content = file.read()
            documents.append(content)
            labels.append(name)  # Replace with the appropriate label extraction logic

# Step 2: Preprocess the text data
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)

    # Add more preprocessing steps if needed

    return text

load_documents()
preprocessed_documents = [preprocess_text(document) for document in documents]

# Step 3: Feature Selection

# Baseline - I (TF)
tf_features_manual = []
for document in preprocessed_documents:
    term_frequency = {}
    total_terms = len(document.split())
    for term in document.split():
        if term in term_frequency:
            term_frequency[term] += 1
        else:
            term_frequency[term] = 1

    tf = {term: term_frequency[term] / total_terms for term in term_frequency}
    tf_features_manual.append(tf)

# Baseline - II (TF-IDF)
tfidf_features_manual = []
total_documents = len(preprocessed_documents)
for document in preprocessed_documents:
    term_frequency = {}
    total_terms = len(document.split())
    for term in document.split():
        if term in term_frequency:
            term_frequency[term] += 1
        else:
            term_frequency[term] = 1

    tf = {term: term_frequency[term] / total_terms for term in term_frequency}

    idf = {}
    for term in tf:
        documents_with_term = sum(1 for doc in preprocessed_documents if term in doc)
        idf[term] = math.log(total_documents / documents_with_term)

    tfidf = {term: tf[term] * idf[term] for term in tf}
    tfidf_features_manual.append(tfidf)

# Step 4: Clustering with K-means

# Baseline - I (TF)
kmeans_baseline_tf_manual = KMeans(n_clusters=5, random_state=42, n_init=10).fit(list(tf_features_manual))

# Baseline - II (TF-IDF)
kmeans_baseline_tfidf_manual = KMeans(n_clusters=5, random_state=42, n_init=10).fit(list(tfidf_features_manual))

# Step 5: Evaluate Clustering Results (Purity Calculation)

def calculate_purity(labels, clusters):
    cluster_labels = {}
    for i in range(len(clusters)):
        cluster_label = labels[i]
        if clusters[i] in cluster_labels:
            cluster_labels[clusters[i]].append(cluster_label)
        else:
            cluster_labels[clusters[i]] = [cluster_label]

    total_documents = len(labels)
    purity = 0

    for cluster in cluster_labels.values():
        class_counts = Counter(cluster)
        most_common_class = class_counts.most_common(1)[0][1]
        purity += most_common_class

    purity /= total_documents
    return purity

#
# Step 6: Word2Vec Embedding

# Train Word2Vec model on the documents
word2vec_model = Word2Vec([document.split() for document in documents], min_count=1)

word2vec_embeddings = []
for document in preprocessed_documents:
    word_embeddings = []
    for word in document.split():
        if word in word2vec_model.wv:
            word_embeddings.append(word2vec_model.wv[word])
    document_embedding = np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(word2vec_model.vector_size)
    word2vec_embeddings.append(document_embedding)

kmeans_word2vec = KMeans(n_clusters=5, random_state=42, n_init=10).fit(word2vec_embeddings)

# Step 8: Evaluate Word2Vec Clustering Results

# Evaluate purity for Word2Vec
purity_word2vec = calculate_purity(labels, kmeans_word2vec.labels_)
print("Purity for Word2Vec:", purity_word2vec)
