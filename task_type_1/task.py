import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import numpy as np

class DetectSpamModel:
    def __init__(self, train_path, test_path):
        """
        Initialize the DetectSpamModel with paths to training and testing data.
        
        Parameters:
            train_path (str): Path to the training data CSV file.
            test_path (str): Path to the testing data CSV file.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    def load_data(self):
        """Load training and testing data from CSV files."""
        X_train = pd.read_csv(self.train_path)['email']
        y_train = pd.read_csv(self.train_path)['label']
        X_test = pd.read_csv(self.test_path)['email']
        y_test = pd.read_csv(self.test_path)['label']
        
        return X_train, y_train, X_test, y_test

    def preprocess_data(self, X_train, y_train, X_test):
        """Preprocess the data by handling missing values and balancing the dataset."""
        # Fill missing values
        X_train = X_train.fillna("")
        y_train = y_train.fillna("")
        X_test = X_test.fillna("")

        # Combine training data into a DataFrame
        train_data = pd.DataFrame({'text': X_train, 'label': y_train})

        # Separate 'ham' and 'spam'
        ham = train_data[train_data['label'] == 'ham']
        spam = train_data[train_data['label'] == 'spam']

        # Oversample 'spam' class
        spam_oversampled = resample(spam, 
                                    replace=True,     
                                    n_samples=len(ham),    
                                    random_state=42)

        # Combine back into a balanced DataFrame
        train_data_balanced = pd.concat([ham, spam_oversampled])

        # Fit and transform the balanced dataset
        X_resampled = self.vectorizer.fit_transform(train_data_balanced['text'])
        y_resampled = train_data_balanced['label'].map({'ham': 0, 'spam': 1})

        # Transform the test set
        X_test_vectorized = self.vectorizer.transform(X_test)

        return X_resampled, y_resampled, X_test_vectorized

    def train_models(self, X_resampled, y_resampled):
        """Train Logistic Regression and Random Forest models."""
        self.lr_model.fit(X_resampled, y_resampled)
        self.rf_model.fit(X_resampled, y_resampled)

    def predict_and_evaluate(self, X_test_vectorized, y_test):
        """Predict using the trained models and evaluate their performance."""
        # Set a threshold for classification
        threshold = 0.36
        y_test_prob = (self.lr_model.predict_proba(X_test_vectorized)[:, 1] + 
                       self.rf_model.predict_proba(X_test_vectorized)[:, 1]) / 2

        # Assign predictions based on the threshold
        y_test_pred = np.where(y_test_prob > threshold, 'spam', 'ham')

        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print("Test Accuracy:", test_accuracy)

        # Display confusion matrix
        print("Confusion Matrix (Test):")
        print(confusion_matrix(y_test, y_test_pred))

        # Output classification report
        print("Classification Report:")
        print(classification_report(y_test, y_test_pred))

    def run(self):
        """Execute the full pipeline: load data, preprocess, train, and evaluate."""
        # Load data
        X_train, y_train, X_test, y_test = self.load_data()

        # Preprocess data
        X_resampled, y_resampled, X_test_vectorized = self.preprocess_data(X_train, y_train, X_test)

        # Train models
        self.train_models(X_resampled, y_resampled)

        # Predict and evaluate
        self.predict_and_evaluate(X_test_vectorized, y_test)


spam_detector = DetectSpamModel('/home/xlordplay/test_tasks/task_type_1/train.csv', '/home/xlordplay/test_tasks/task_type_1/test.csv')
spam_detector.run()


"""
Test Accuracy: 0.980269989615784
Confusion Matrix (Test):
[[2427   40]
 [  17  405]]
Classification Report:
              precision    recall  f1-score   support

         ham       0.99      0.98      0.99      2467
        spam       0.91      0.96      0.93       422

    accuracy                           0.98      2889
   macro avg       0.95      0.97      0.96      2889
weighted avg       0.98      0.98      0.98      2889
"""
