import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec


#initialze the neccessary variables here 

# Initialize the arrays
documents = []
labels = []

# define the array for the preprocessed documents with its labels 
preprocessed_documents = []
preprocessed_labels = []


# Define the paths to the folders
doc_path = "Doc50"
labels_folder = "Doc50 GT"

# stores all the stop words in a variable 
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#storing the function to access it efficiently
vectorizer = TfidfVectorizer()


# to lower the document 
def do_lower_case(doc):
    lower_doc = doc.lower()
    return lower_doc

# function for lemmetizaion
def do_lemma(tokens , stop_words):
    prep_tokens = []
    for token in tokens:
        if token not in stop_words:
            prep_token = lemmatizer.lemmatize(token)
            prep_tokens.append(prep_token)
    return prep_tokens

# function for preprocessing the documents 
def preprocessing(documents, labels):
    pre_doc = []
    pre_labels = []
    for document, label in zip(documents, labels):
        # Convert to lowercase
        document = do_lower_case(document) 

        # Remove non-alphanumeric characters
        document = re.sub(r'\W+', ' ', document)

        # Tokenization
        tokens = word_tokenize(document)

        # Remove stopwords and perform lemmatization
        pre_tokens = do_lemma(tokens, stop_words)

        # Join the tokens back into a document
        pre_pro_document = ' '.join(pre_tokens)

        # Add the preprocessed document and label to the respective lists
        pre_doc.append(pre_pro_document)
        pre_labels.append(label)
    return pre_doc , pre_labels



# function to Evaluate clustering results using purity
def Baseline_purity(labels, true_labels):
    cluster_purities = []
    u_clusters = np.unique(labels)

    # Extract the numeric part from true_labels and convert it to integers
    true_labels_numeric = [int(label[1:]) for label in true_labels]

    for cluster in u_clusters:
        cluster_labels = np.array(true_labels_numeric)[labels == cluster]
        if len(cluster_labels) > 0:
            common_label = np.bincount(cluster_labels).argmax()
            cluster_purities.append(np.sum(cluster_labels == common_label) / len(cluster_labels))

    return np.mean(cluster_purities)

# function to calculate the purity of Word2vec 
def word2Vec_purity(labels, true_labels):
    cluster_purities = []
    u_clusters = np.unique(labels)

    # Extract the numeric part from true_labels and convert it to integers
    true_labels_numeric = [int(label[1:]) for label in true_labels]

    # Ensure labels and true_labels_numeric have the same length
    labels = labels[:len(true_labels_numeric)]

    for cluster in u_clusters:
        cluster_labels = np.array(true_labels_numeric)[np.array(labels) == cluster]
        if len(cluster_labels) > 0:
            common_label = np.bincount(cluster_labels).argmax()
            cluster_purities.append(np.sum(cluster_labels == common_label) / len(cluster_labels))

    return np.mean(cluster_purities)

# function to calculate word2Vec and its purity 
def word2Vec(pre_documents , pre_labels):
    # Train Word2Vec model
    sentences = [word_tokenize(doc) for doc in pre_documents]
    word2vec_model = Word2Vec(sentences, vector_size=100, min_count=5)

    # Get word embeddings
    word_embeddings = word2vec_model.wv

    # Perform clustering on word embeddings
    kmeans_word2vec = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans_word2vec.fit(word_embeddings.vectors)
    labels_word2vec = kmeans_word2vec.labels_

    # Calculate purity for Word2Vec-based clustering
    purity_word2vec = word2Vec_purity(labels_word2vec, pre_labels)
    return purity_word2vec


# Load the all documents along with their labels 
for filename in os.listdir(doc_path):
    with open(os.path.join(doc_path, filename), "r") as file:
        document = file.read()
        documents.append(document)

    # load the label of the documents 
    for label_folder in os.listdir(labels_folder):
        if filename in os.listdir(os.path.join(labels_folder, label_folder)):
            labels.append(label_folder)
            break

preprocessed_documents , preprocessed_labels = preprocessing(documents , labels)

# Convert preprocessed documents to TF-IDF features
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)

# Baseline I: Clustering using all TF-based features
kmeans_baseline1 = KMeans(n_clusters=5, random_state=42,n_init=10)
kmeans_baseline1.fit(tfidf_matrix)
labels_baseline1 = kmeans_baseline1.labels_

# Baseline II: Clustering using TF*IDF features and DF filtering
tfidf_vectorizer = TfidfVectorizer(min_df=2)
tfidf_matrix_baseline2 = tfidf_vectorizer.fit_transform(preprocessed_documents)

kmeans_baseline2 = KMeans(n_clusters=5, random_state=42,n_init=10)
kmeans_baseline2.fit(tfidf_matrix_baseline2)
labels_baseline2 = kmeans_baseline2.labels_

# Calculate purity for baseline I
print("Purity for Baseline I :", Baseline_purity(labels_baseline1, preprocessed_labels))

# Calculate purity for baseline II
print("Purity for Baseline II:", Baseline_purity(labels_baseline2, preprocessed_labels))

#Calculate purity for word2vec
print("Purity for Word2Vec :", word2Vec(preprocessed_documents , preprocessed_labels))





