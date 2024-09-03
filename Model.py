import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Sample dataset
data = {
    'text': [
        "I love this product!", 
        "This is terrible.", 
        "Great experience!", 
        "Awful service.", 
        "Highly recommended!",
        "Never buying again.",
        "Amazing quality!",
        "Disappointing results.",
        "Excellent customer support!",
        "Waste of money."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text to numerical features
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_vectorized)

# Print the accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Save the model and vectorizer
joblib.dump(clf, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

# Function to predict sentiment of new text
def predict_sentiment(text):
    vectorized_text = vectorizer.transform([text])
    prediction = clf.predict(vectorized_text)
    return "Positive" if prediction[0] == 1 else "Negative"

# Test the model with a new sentence
print(predict_sentiment("I really enjoyed using this product!"))