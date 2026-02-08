import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1️⃣ Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# 2️⃣ Standardize column names
df = df.rename(columns={"review": "text", "sentiment": "label"})

# 3️⃣ Convert sentiment labels to numeric
df["label"] = df["label"].map({"positive": 1, "negative": 0})

# 4️⃣ Drop missing values
df = df.dropna()

# 5️⃣ Split data
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7️⃣ Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 8️⃣ Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 9️⃣ Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ sentiment_model.pkl and vectorizer.pkl created successfully!")