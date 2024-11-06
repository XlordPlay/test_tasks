import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField
from wtforms.validators import DataRequired

# Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Form class to handle CSV file upload and search query input
class UploadForm(FlaskForm):
    file = FileField('CSV file', validators=[DataRequired()])  # File upload field
    search_query = StringField('Search Query', validators=[DataRequired()])  # Search input field
    submit = SubmitField('Search')  # Submit button to trigger search

# Global variables to store documents, labels, and TF-IDF vectors
documents = []
labels = []
tfidf_vectorizer = TfidfVectorizer()  # TF-IDF Vectorizer to convert text to vectors
tfidf_matrix = None  # Store the TF-IDF matrix for all documents
top_n = 5  # Number of top relevant documents to return


def evaluate_search_results(true_indices, retrieved_indices):
    true_positives = len(set(true_indices).intersection(set(retrieved_indices)))  # True positives (relevant and retrieved)
    
    # Precision: How many of the retrieved docs are relevant
    precision = true_positives / len(retrieved_indices) if retrieved_indices else 0
    
    # Recall: How many of the relevant docs were retrieved
    recall = true_positives / len(true_indices) if true_indices else 0
    
    # F1 score: Harmonic mean of precision and recall
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score  # Return calculated metrics


# Function to determine the cosine similarity threshold based on the query length
def get_threshold(query_length):
    if query_length <= 3 or (query_length > 3 and query_length < 10):
        return 0.1  # Low threshold for very short queries
    elif query_length >= 10 and query_length <= 20:
        return 0.3  # Medium threshold for average-length queries
    elif query_length > 20 and query_length <= 30:
        return 0.4  # Higher threshold for long queries
    else:
        return 0.5  # Very high threshold for very long queries

@app.route('/', methods=['GET', 'POST'])
def index():
    global documents, labels, tfidf_matrix
    form = UploadForm()  # Instantiate the form object
    results = []  
    metrics = {} 

    # Check if the form is valid and was submitted
    if form.validate_on_submit():
        file = form.file.data  
        df = pd.read_csv(file)  

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        # Extract text and labels from the DataFrame
        documents = df['text'].tolist()
        labels = df['labels'].tolist()  # Optional: If you have labels for documents
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)  # Transform documents into TF-IDF matrix

        # Process the search query and convert it into a vector
        search_query = form.search_query.data  # Get the search query from the form
        query_vector = tfidf_vectorizer.transform([search_query])  # Convert search query into a vector
        
        # Compute cosine similarity between the query vector and all document vectors
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get the threshold value based on the length of the search query
        threshold = get_threshold(len(search_query.split()))

        # Retrieve the indices of the top-N documents based on similarity
        related_docs_indices = cosine_similarities.argsort()[-top_n:][::-1]
        results = [(documents[i], cosine_similarities[i], labels[i]) for i in related_docs_indices]  # Get relevant results

        # Identify relevant documents based on the threshold for cosine similarity
        true_indices = [i for i in range(len(cosine_similarities)) if cosine_similarities[i] > threshold]
        retrieved_indices = related_docs_indices.tolist()  # List of indices of retrieved documents

        # Evaluate the search results based on precision, recall, and F1 score
        if true_indices:
            precision, recall, f1_score = evaluate_search_results(true_indices, retrieved_indices)
            metrics = {  
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }

    # Render the results on the HTML page
    return render_template('index.html', form=form, results=results, metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
