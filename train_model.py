import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample email dataset
data = {
    "message": [
        "Congratulations you won a free prize",
        "Claim your free cash reward now",
        "You have won a lottery",
        "Click here to claim your prize",
        "Win money now limited offer",
        "Free entry in our competition",
        "Urgent claim your reward today",
        "You are selected for a cash prize",
        "Get a free gift card now",
        "Earn money quickly from home",
        "Your account won a bonus",
        "Special offer buy now",
        "You have been selected as a winner",
        "Claim your free mobile phone",
        "Get a loan instantly apply now",
        "Congratulations claim your gift",
        "Limited time offer click now",
        "You won free tickets",
        "Exclusive cash reward waiting",
        "Send details to claim prize",

        "Hello, how are you?",
        "Please call me when you are free",
        "Are we meeting today?",
        "I will reach college at ten",
        "Please send me the class notes",
        "Your assignment is due tomorrow",
        "Thank you for your help",
        "Can you share the meeting details?",
        "I am going to the library",
        "See you in class tomorrow",
        "Please check your email",
        "The lecture starts at nine",
        "I have completed my homework",
        "Let us meet after college",
        "Please bring your notebook",
        "I will call you later",
        "The project presentation is tomorrow",
        "Have you finished the assignment?",
        "Good morning, how are you?",
        "Please send the document"
    ],
    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",

        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

# Create the machine learning model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print(f"Test accuracy on sample data: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
joblib.dump(model, "spam_model.joblib")

print("\nModel saved as spam_model.joblib")