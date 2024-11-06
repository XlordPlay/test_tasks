import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import numpy as np

#Load the training and testing data from CSV files (replace for u path)
X_train = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/train.csv')['email']
y_train = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/train.csv')['label']
X_test = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/test.csv')['email']
y_test = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/test.csv')['label']

#Handle missing values by filling them with an empty string
X_train = X_train.fillna("")
y_train = y_train.fillna("")
X_test = X_test.fillna("")
y_test = y_test.fillna("")

#Combine the training data into a single DataFrame for easier manipulation
train_data = pd.DataFrame({'text': X_train, 'label': y_train})

#Separate the data into 'ham' and 'spam' categories for balancing
ham = train_data[train_data['label'] == 'ham']
spam = train_data[train_data['label'] == 'spam']

#Perform oversampling on the 'spam' class to balance the dataset
spam_oversampled = resample(spam, 
                            replace=True,      # Allow replacement to create new samples
                            n_samples=len(ham),    # Match the number of 'ham' samples
                            random_state=42)  # Set seed for reproducibility

#Combine the balanced 'ham' and oversampled 'spam' back into a single DataFrame
train_data_balanced = pd.concat([ham, spam_oversampled])

#Create a TF-IDF vectorizer to convert text data into numerical format
vectorizer = TfidfVectorizer(stop_words='english')

#Fit and transform the balanced training data into TF-IDF format
X_resampled = vectorizer.fit_transform(train_data_balanced['text'])

#Map labels to numerical values: 'ham' -> 0 and 'spam' -> 1
y_resampled = train_data_balanced['label'].map({'ham': 0, 'spam': 1})

#Transform the test data using the same vectorizer
X_test_vectorized = vectorizer.transform(X_test)

#Train a Logistic Regression model with balanced class weights
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_resampled, y_resampled)

#Train a Random Forest Classifier with balanced class weights
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_resampled, y_resampled)

#Adjust the threshold to control sensitivity
threshold = 0.36

#Compute average predicted probabilities for the test set from both models
y_test_prob = (lr_model.predict_proba(X_test_vectorized)[:, 1] + rf_model.predict_proba(X_test_vectorized)[:, 1]) / 2

# Classify as 'spam' if probability exceeds the threshold, otherwise 'ham'
y_test_pred = np.where(y_test_prob > threshold, 'spam', 'ham')

#Calculate the accuracy of the predictions
test_accuracy = accuracy_score(y_test, y_test_pred)  
print("Test Accuracy:", test_accuracy)

#Display the confusion matrix for the test predictions
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))

#Output a detailed classification report including precision, recall, and F1-score
print("Classification Report:")
print(classification_report(y_test, y_test_pred))


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
