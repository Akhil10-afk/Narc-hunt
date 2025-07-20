import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample data
data = {
    'message': [
        'Buy weed now',
        'Special offer on cocaine',
        'Let’s catch up tomorrow',
        'Need MDMA urgently',
        'How are you?',
        'Get your heroin here'
    ],
    'label': [1, 1, 0, 1, 0, 1]  # 1 = drug-related, 0 = normal
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert text to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

# Train model
model = MultinomialNB()
model.fit(X, df['label'])

# Save model and vectorizer
joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("✅ Model trained and saved.")