# Python code for sentiment analysis using machine learning and NLP

import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('airline_reviews.csv')

# Preprocess the data using NLP techniques
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
vectorizer = CountVectorizer(stop_words=stopwords)
X = vectorizer.fit_transform(df['review'].values)
y = df['sentiment'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier on the training set
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the accuracy of the model on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
