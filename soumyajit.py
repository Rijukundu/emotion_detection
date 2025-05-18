import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import re
import string

# Load dataset
data = pd.read_csv('tweet_emotions.csv')



# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['content'] = data['content'].apply(clean_text)

# Features and labels
X = data['content']
y = data['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
print("Model Performance:\n")
print(classification_report(y_test, model.predict(X_test)))

# Predict on user input
while True:
    user_input = input("\nEnter text to detect emotion (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    cleaned_input = clean_text(user_input)
    emotion = model.predict([cleaned_input])[0]
    print(f"Predicted Emotion: {emotion}")
