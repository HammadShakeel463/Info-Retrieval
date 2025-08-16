
# define all the necessary libraries

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
import math
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import glob


# Read the file that comtains all the provided stop words with the name "stopwords.txt"
with open("stopwords.txt", "r") as f:
    stop_words = f.read().split()


#load all the documents from the current directory 
def load_documents(docs_dir):
# Initialize an empty list to hold the documents
    documents = []
    
    # Loop through the range of document numbers 1 to 30
    for i in range(1, 31):
    
    # Construct the file name by appending the document number to '.txt'
        file_name = f"{i}.txt"
        
        # Construct the file path by joining the docs_dir and file_name
        file_path = os.path.join(docs_dir, file_name)

    # Read the content of the file
        with open(file_path, 'r') as f:
            content = f.read()
            documents.append(content)
    return documents


# function for the tokonization of all the documents
def tokenize_documents(documents, stemmer, stop_words):
    # Create an empty list to store tokenized documents
    tokenized_documents = []
    
    # Loop through each document in the input list
    for document in documents:
    
    # Tokenize the document by converting all text to lowercase and splitting it into individual words
        tokenized = word_tokenize(document.lower())
        
        # Apply stemming and remove stop words from the tokens
        tokenized = [stemmer.stem(word) for word in tokenized if word not in stop_words]
        
        # Add the tokenized document to the list of tokenized documents
        tokenized_documents.append(tokenized)
    return tokenized_documents


#creating the vocablary of all the terms appeared in the documents 
def create_vocabulary(tokenized_documents):
    vocabulary = list(set([word for document in tokenized_documents for word in document]))
    
    return vocabulary


#creating the inverted index of all the terms 
def create_inverted_index(tokenized_documents):
    # Create an empty dictionary to store the inverted index
    inverted_index = {}
    
    # Loop over each document in the list of tokenized documents
    for i, document in enumerate(tokenized_documents):
        for term in document:
        
        # If the term is already in the inverted index, append the document index to its list of postings
            if term in inverted_index:
                inverted_index[term].append(i)
            
            # Otherwise, add the term to the inverted index with the current document index as its first posting
            else:
                inverted_index[term] = [i]
    return inverted_index


# creating the positional indexes
def create_positional_index(tokenized_documents):
    # create an empty dictionary to store the positional index
    positional_index = {}
    
    
    for i, document in enumerate(tokenized_documents):
        for j, term in enumerate(document):
            
             # if the term is already in the positional index, append the current document and term position to its postings list
            if term in positional_index:
                positional_index[term][i].append(j)
            
            # if the term is not yet in the positional index, create a new entry for it with the current document and term position

            else:
                positional_index[term] = {i: [j] for i in range(len(tokenized_documents))}
    return positional_index

# now we calculate the tf-idf value of all the documents 

# Calculate document frequency (df) for each term in the vocabulary
def calculate_df(vocabulary, inverted_index):
    df = {}
    for term in vocabulary:
        df[term] = len(inverted_index[term])
    return df


# Calculate inverse document frequency (idf) for each term in the vocabulary
def calculate_idf(vocabulary, df, num_documents):
    idf = {}
    for term in vocabulary:
        idf[term] = math.log(num_documents / df[term])
    return idf
    
# Calculate term frequency (tf) for each document in the tokenized_documents list
def calculate_tf(tokenized_documents, vocabulary):
    tf = {}
    for i, document in enumerate(tokenized_documents):
        tf[i] = {}
        for term in vocabulary:
            tf[i][term] = document.count(term)
    return tf

# Calculate tf-idf score for each term in each document
def calculate_tf_idf(tf, idf, vocabulary):
    tf_idf = {}
    for i in tf:
        tf_idf[i] = {}
        for term in vocabulary:
            tf_idf[i][term] = tf[i][term] * idf[term]
    return tf_idf

    

# Calculate tf-idf score for each term in the query
def calculate_query_tf_idf(query, vocabulary, idf):
    query_tf_idf = {}
    for term in vocabulary:
        query_tf_idf[term] = query.count(term) * idf[term]
    return query_tf_idf

def calculate_cosine_similarity(tf_idf, query_tf_idf, vocabulary,tokenized_documents):
    # initialize an empty dictionary to store the cosine similarity score for each document
    similarities = {}
        
    for i, document in enumerate(tokenized_documents):
        # calculate the dot product of the document's TF-IDF vector and the query's TF-IDF vector
        dot_product = sum(tf_idf[i][term] * query_tf_idf[term] for term in vocabulary)
        
        # calculate the Euclidean length of the document's TF-IDF vector
        document_length = math.sqrt(sum(tf_idf[i][term]**2 for term in vocabulary))
        
        # calculate the Euclidean length of the query's TF-IDF vector
        query_length = math.sqrt(sum(query_tf_idf[term]**2 for term in vocabulary))
        
         # check if both document_length and query_length are not zero
        if document_length != 0 and query_length != 0: 
        
        # calculate the cosine similarity score for the document and store it in the similarities dictionary
            cosine_similarity = dot_product / (document_length * query_length)  
        else:
        # if either document_length or query_length is zero, set the cosine similarity score to zero
            cosine_similarity = 0
        similarities[i] = cosine_similarity
    return similarities


def filter_documents(similarities, alpha):
    # Sort the documents by their similarity score in descending order.
    ranked_documents = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    
    # Create a filtered_documents list that includes the documents with similarity score greater than or equal to alpha.
    filtered_documents = [(doc_id, score) for doc_id, score in ranked_documents if score >= alpha]
    return filtered_documents


#this is the class for the graphical user interface 
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set up the main window
        self.setWindowTitle("Search Engine")
        self.setGeometry(100, 100, 800, 600)

        # Set up the central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Set up the layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Add a label for the query input
        query_label = QLabel("Enter your query:")
        main_layout.addWidget(query_label)

        # Add a text input field for the query
        self.query_input = QLineEdit()
        main_layout.addWidget(self.query_input)

        # Add a button to trigger the search
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.search)
        main_layout.addWidget(search_button)

        # Add a label for the search results
        results_label = QLabel("Search results:")
        main_layout.addWidget(results_label)

        # Add a text edit for the search results
        self.results_output = QTextEdit()
        main_layout.addWidget(self.results_output)

        # Set up the search engine
        self.docs_dir = "./docs"
        self.stemmer = SnowballStemmer("english")
        self.documents = load_documents(self.docs_dir)
        self.tokenized_documents = tokenize_documents(self.documents, self.stemmer, stop_words)
        self.vocabulary = create_vocabulary(self.tokenized_documents)
        self.inverted_index = create_inverted_index(self.tokenized_documents)
        self.positional_index = create_positional_index(self.tokenized_documents)
        self.num_documents = len(self.tokenized_documents)
        self.df = calculate_df(self.vocabulary, self.inverted_index)
        self.idf = calculate_idf(self.vocabulary, self.df, self.num_documents)
        self.tf = calculate_tf(self.tokenized_documents, self.vocabulary)
        self.tf_idf = calculate_tf_idf(self.tf, self.idf, self.vocabulary)
        
  # This part conatians the styling of the graphical user interface 
        self.setStyleSheet('''
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333333;
            }

            QLineEdit {
                border: 2px solid #888888;
                border-radius: 5px;
                font-size: 16px;
                padding: 5px;
            }

            QTextEdit {
                border: 2px solid #888888;
                border-radius: 5px;
                font-size: 16px;
                padding: 5px;
            }

            QPushButton {
                background-color: #0066cc;
                border: none;
                border-radius: 5px;
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }

            QListWidget {
                border: 2px solid #888888;
                border-radius: 5px;
                font-size: 16px;
                padding: 5px;
            }

            QListWidget::item:selected {
                background-color: #0066cc;
                color: #ffffff;
            }
        ''')


    # this part performs all the calculations and call all the functions
    def search(self):
        # Get the query from the input field
        query = self.query_input.text()

        # Tokenize and stem the query
        query_tokenized = [self.stemmer.stem(word.lower()) for word in word_tokenize(query) if word not in stop_words]

        # Calculate the query tf-idf
        query_tf_idf = calculate_query_tf_idf(query_tokenized, self.vocabulary, self.idf)

        # Set the filtering threshold
        alpha = 0.0005

        # Calculate the cosine similarity for each document
        similarities = calculate_cosine_similarity(self.tf_idf, query_tf_idf, self.vocabulary, self.tokenized_documents)

        # Filter the documents by similarity score
        filtered_documents = filter_documents(similarities, alpha)

        # Display the search results
        self.results_output.clear()
        if len(filtered_documents) > 0 : 
            for doc_id, score in filtered_documents:
                if(score > 0):
                    self.results_output.insertPlainText(f"Document {doc_id+1}: {score:.4f}\n")
        else:
            self.results_output.insertPlainText(f"NO DOCUMENTS\n")

# main function that create the gui object 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

