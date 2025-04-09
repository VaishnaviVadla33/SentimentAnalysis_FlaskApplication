import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle
import nltk

# Download stopwords if not already present
nltk.download("stopwords")

# Sample dataset
data = {
    "review": ["This movie is amazing!", "I hated this movie.", "Fantastic storyline!", "Worst movie ever."],
    "sentiment": ["positive", "negative", "positive", "negative"]
}

df = pd.DataFrame(data)

# Train model
vectorizer = CountVectorizer(stop_words="english")
model = make_pipeline(vectorizer, MultinomialNB())

X = df["review"]
y = df["sentiment"]
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
