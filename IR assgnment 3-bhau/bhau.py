import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec


def load(directory):
        # Load the all documents 
    documents = [] 
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r") as file:
            document = file.read()
            documents.append(document)
    return documents

def load_label(directory , labels_path):
    # Load the labels after reading all the documents
    labels = []
    for filename in os.listdir(directory):
        for label_folder in os.listdir(labels_path):
            if filename in os.listdir(os.path.join(labels_path, label_folder)):
                labels.append(label_folder)
                break
    return labels 

def calculate_baseline(vectorizer , p_documents):
    tfidf_matrix = vectorizer.fit_transform(p_documents)

    # Baseline I: Clustering using all TF-based features
    kmeans_baseline1 = KMeans(n_clusters=5, random_state=42,n_init=10)
    kmeans_baseline1.fit(tfidf_matrix)

    # Baseline II: Clustering using TF*IDF features and DF filtering
    tfidf_vectorizer = TfidfVectorizer(min_df=2)
    tfidf_matrix_baseline2 = tfidf_vectorizer.fit_transform(p_documents)
    kmeans_baseline2 = KMeans(n_clusters=5, random_state=42,n_init=10)
    kmeans_baseline2.fit(tfidf_matrix_baseline2)

    labels_baseline1 = kmeans_baseline1.labels_
    labels_baseline2 = kmeans_baseline2.labels_

    return labels_baseline1 , labels_baseline2

def calculate_purity(labels, tlabels , num):

    if num:
        cluster_p = []
        unique_clusters = np.unique(labels)

        # Extract the numeric part from true_labels and convert it to integers
        true_labels_numeric = [int(label[1:]) for label in tlabels]

        # Ensure labels and true_labels_numeric have the same length
        labels = labels[:len(true_labels_numeric)]

        for cluster in unique_clusters:
            cluster_labels = np.array(true_labels_numeric)[np.array(labels) == cluster]
            if len(cluster_labels) > 0:
                most_common_label = np.bincount(cluster_labels).argmax()
                cluster_p.append(np.sum(cluster_labels == most_common_label) / len(cluster_labels))

    else:
        cluster_p = []
        unique_clusters = np.unique(labels)

        # Extract the numeric part from true_labels and convert it to integers
        true_labels_numeric = [int(label[1:]) for label in tlabels]

           # Extract the numeric part from true_labels and convert it to integers
        true_labels_numeric = [int(label[1:]) for label in tlabels]

        for cluster in unique_clusters:
            cluster_labels = np.array(true_labels_numeric)[labels == cluster]
            if len(cluster_labels) > 0:
                most_common_label = np.bincount(cluster_labels).argmax()
                cluster_p.append(np.sum(cluster_labels == most_common_label) / len(cluster_labels))

    return np.mean(cluster_p)


# Define the paths to the folders
directory = "Doc50"
labels_path = "Doc50 GT"

# Initialize the arrays
documents = load(directory)
labels = load_label(directory , labels_path)

# Preprocessing steps
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

p_documents = []
p_labels = []

for document, label in zip(documents, labels):
    # Convert to lowercase
    document = document.lower()

    # Remove non-alphanumeric characters
    document = re.sub(r'\W+', ' ', document)

    # Tokenization
    tokens = word_tokenize(document)

    # Remove stopwords and perform lemmatization
    preprocessed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    # Join the tokens back into a document
    preprocessed_document = ' '.join(preprocessed_tokens)

    # Add the preprocessed document and label to the respective lists
    p_documents.append(preprocessed_document)
    p_labels.append(label)

# Convert preprocessed documents to TF-IDF features
vectorizer = TfidfVectorizer()

baseline1_label , baseline2_label = calculate_baseline(vectorizer , p_documents)
# Calculate purity for baseline I
purity_baseline1 = calculate_purity(baseline1_label, p_labels , 0)
print("Purity (Baseline I):", purity_baseline1)

# Calculate purity for baseline II
purity_baseline2 = calculate_purity(baseline2_label, p_labels, 0)
print("Purity (Baseline II):", purity_baseline2)



# Train Word2Vec model
sentences = [word_tokenize(doc) for doc in p_documents]
word2vec_model = Word2Vec(sentences, vector_size=100, min_count=5)

# Get word embeddings
word_embeddings = word2vec_model.wv

# Perform clustering on word embeddings
kmeans_word2vec = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_word2vec.fit(word_embeddings.vectors)
labels_word2vec = kmeans_word2vec.labels_

# Calculate purity for Word2Vec-based clustering
word2vec_purity = calculate_purity(labels_word2vec, p_labels , 1)
print("Purity (Word2Vec):", word2vec_purity)





