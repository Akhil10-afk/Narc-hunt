import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Input message
message = input("Enter message: ")
message_vec = vectorizer.transform([message])
prediction = model.predict(message_vec)

# Output result
if prediction[0] == 1:
    print("🚨 Drug-related message detected!")
else:
    print("✅ Message is clean.")
