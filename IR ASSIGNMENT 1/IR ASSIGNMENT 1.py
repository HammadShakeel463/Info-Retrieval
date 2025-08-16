# At first we import all the necessary libraries

from nltk.stem import SnowballStemmer
import re
from PyQt5 import QtWidgets, QtGui, QtCore
import boolean

# Create a Snowball stemmer for English language
stemmer = SnowballStemmer("english")

# Read the file that comtains all the provided stop words with the name "stopwords.txt"
with open("stopwords.txt", "r") as f:
    stop_words = f.read().split()



# Preprocess the queries by tokenizing, and stemming
def preprocess_q(text):
    text = text.lower()
    return stemmer.stem(text)

# Preprocess the text by tokenizing, removing stop words, and stemming
def preprocess(text):
    # Convert the text to lowercase
    text = text.lower()
    # Tokenize by splitting on whitespace and punctuation
    tokens = re.findall(r"\b\w+\b", text)
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    # Stem tokens
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# Create an inverted index from a list of documents
def create_inverted_index(docs):
    inverted_index = {}
    
    # iterate over each document for creating its inverted index
    for doc_id, doc in enumerate(docs):
        tokens = preprocess(doc)
        for position, token in enumerate(tokens):
            if token not in inverted_index:
                inverted_index[token] = {}
            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = []
            inverted_index[token][doc_id].append(position)
            
    return inverted_index



# below function will handle all the queries with the format (T1 AND T2 OR T3)
def evaluate_query1(query, inverted_index, docs):

    # Initialize empty lists for the terms and operators
    terms = []
    operators = []
    query = query.split()
    # Loop through tokens and separate terms and operators
    for token in query:
        if token in ("AND", "OR", "NOT"):
            operators.append(token)
        else:
            terms.append(token)
    #print(terms)
    
    words = []
    
    for term in terms:
        words.append(preprocess_q(term))
    # Initialize empty list for the results of each term
    term_results = []
    #print(words)
    
    # Loop through terms and find matching documents
    for term in words:
        # Find documents containing the term
        term_docs = inverted_index.get(term, [])
        term_results.append(term_docs)
        #print(term_docs)
    
    # Combine term results using boolean operators
    for operator in operators:
        if operator == "AND":
            # Take the intersection of the results of the previous two terms
            term_results.append(set(term_results[-2]).intersection(set(term_results[-1])))
        elif operator == "OR":
            # Take the union of the results of the previous two terms
            term_results.append(set(term_results[-2]).union(set(term_results[-1])))
        elif operator == "NOT":
            num_doc = []
            for i in range(len(docs)):
                num_doc.append(i) 
            print(set(term_results[-1]))
            # Find all documents not containing the following term
            term_results.append(set(num_doc).difference(set(term_results[-1])))
    
    # Return the final set of matching documents
    return term_results[-1]



# this function will handle the positional index quries with the format (X Y / num) where num is the distance of two words

def evaluate_query(query, inverted_index, docs):
    # preprocess the query
    query_terms = query.split()
   
    # handle proximity queries
    if len(query_terms) == 4 and query_terms[2] == '/' and query_terms[3].isdigit():
        # extract query parameters
	
        distance = int(query_terms[3])
        query_terms = preprocess(query)
        # storing the first term which is X in term1 and second term with is y in term2
        term1 = query_terms[0]
        term2 = query_terms[1]

        # get posting lists for both terms
        postings1 = inverted_index.get(term1, [])
        postings2 = inverted_index.get(term2, [])
        #print(postings1)
        # merge the posting lists and calculate proximity
        results = []
        i, j = 0, 0
        
        # iterating over the documnts and its posting lists to find all the document that contains the both the terms with the given distance 
        while i < len(postings1) and j < len(postings2):
            doc1 = list(postings1.keys())[i]
            doc2 = list(postings2.keys())[j]
            positions1 = list(postings1.values())[i]
            positions2 = list(postings2.values())[j]
            #print(positions1)
            #print(doc1)
            if doc1 == doc2:
                # calculate proximity between positions
                for pos1 in positions1:
                    for pos2 in positions2:
                        if abs(pos1 - pos2) == distance:
                            results.append(doc1)
                            break
            if doc1 < doc2:
                i += 1
            else:
                j += 1
	#return the result
        return results

    # handle boolean queries
    else:
    
        return evaluate_query1(query , inverted_index , docs)




# GUI code

class BooleanRetrievalGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Boolean Retrieval Model')

        # Create label and text box for document input
        doc_label = QtWidgets.QLabel('Enter documents (one per line):')
        self.doc_textbox = QtWidgets.QTextEdit()
        self.doc_textbox.setFixedHeight(100)
        doc_layout = QtWidgets.QVBoxLayout()
        doc_layout.addWidget(doc_label)
        doc_layout.addWidget(self.doc_textbox)

        # Create label and text box for query input
        query_label = QtWidgets.QLabel('Enter query (e.g. t1 AND t2 OR t3) OR (X Y / num ):')
        self.query_entry = QtWidgets.QLineEdit()
        query_layout = QtWidgets.QVBoxLayout()
        query_layout.addWidget(query_label)
        query_layout.addWidget(self.query_entry)

        # Create button to process documents and query
        self.process_button = QtWidgets.QPushButton('Process')
        self.process_button.clicked.connect(self.process_query)
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.addWidget(self.process_button)

        # Create list box to display matching documents
        self.doc_listbox = QtWidgets.QListWidget()
        doc_layout.addWidget(self.doc_listbox)

        # Create main layout
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(doc_layout)
        main_layout.addLayout(query_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        # Set style sheet
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

    def process_query(self):
        # Get documents from text box
        self.doc_listbox.clear()
        docs_name = self.doc_textbox.toPlainText().strip().split("\n")
        docs = []
        for i in range(len(docs_name)):
            with open(docs_name[i], "r") as f:
                doc = f.read()
            docs.append(doc)

        # Create inverted index
        inverted_index = create_inverted_index(docs)

        # Evaluate query on inverted index
        query = self.query_entry.text()
        matching_docs = evaluate_query(query, inverted_index, docs)

        # Add matching documents to list box
        if len(matching_docs) == 0:
            QtWidgets.QMessageBox.information(self, "No Matches", "No documents matched the query.")
        else:
            for doc_id in matching_docs:
                self.doc_listbox.addItem(f"Document {doc_id + 1}")


if __name__ == '__main__':
    # Create application instance
    app = QtWidgets.QApplication([])

    # Create main window instance
    window = BooleanRetrievalGUI()

    # Show the window
    window.show()

    # Run the event loop
    app.exec_()
