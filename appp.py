import streamlit as st
import pandas as pd
import joblib

# ---------------- Load model & vectorizer ----------------
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Sentiment Insights",
    page_icon="😊",
    layout="centered"
)

# ---------------- Header ----------------
st.markdown(
    """
# 😊 Sentiment Insights

### Instantly understand the emotion behind text

Analyze text sentiment using a trained machine learning model.  
You can **type text manually** or **upload a CSV file** for bulk analysis.
"""
)

st.divider()

# ---------------- Accuracy Info ----------------
st.info("📊 Model Accuracy: ~88% (evaluated on test dataset)")

st.divider()

# ---------------- Text Analysis ----------------
st.markdown(
    """
## ✍️ Analyze Text

Enter any sentence, review, or feedback below and click **Analyze Text**
to identify whether the sentiment is **Positive** or **Negative**.
"""
)

user_text = st.text_area(
    "📝 Enter text here",
    placeholder="Example: This app is simple and very easy to use!"
)

if st.button("🔍 Analyze Text"):
    if user_text.strip():
        vec = vectorizer.transform([user_text])
        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec)[0].max()

        if prediction == 1:
            st.success("😊 Sentiment: Positive")
        else:
            st.error("😞 Sentiment: Negative")

        st.write(f"📈 Confidence Score: {confidence:.2f}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

st.divider()

# ---------------- CSV Analysis ----------------
st.markdown(
    """
## 📂 Analyze CSV File

Upload a CSV file to analyze multiple text entries at once.

### CSV requirements:
- The file must contain a column named **`text`**
- Each row should contain one text entry
"""
)

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("❌ CSV file must contain a column named 'text'.")
    else:
        vec = vectorizer.transform(df["text"])
        df["Sentiment"] = model.predict(vec)
        df["Sentiment"] = df["Sentiment"].map({
            1: "Positive 😊",
            0: "Negative 😞"
        })

        st.success("✅ Sentiment analysis completed!")
        st.dataframe(df.head())

        st.download_button(
            label="⬇️ Download Results",
            data=df.to_csv(index=False),
            file_name="sentiment_results.csv",
            mime="text/csv"
        )

st.divider()

# ---------------- Footer ----------------
st.caption(
    "🚀 Built with Python & Streamlit • Supports real-time and batch sentiment analysis"
)