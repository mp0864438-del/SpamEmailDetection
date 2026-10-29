
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Spam Email Detection")
st.write("Machine Learning Based Email Classifier")
st.divider()

# Load trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "spam_model.joblib"
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("Model load झाला नाही. spam_model.joblib तपासा.")
    st.stop()

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Dashboard metrics
total = len(st.session_state.history)
spam_count = sum(
    item["Result"] == "SPAM"
    for item in st.session_state.history
)

col1, col2, col3 = st.columns(3)
col1.metric("Emails Tested", total)
col2.metric("Spam Detected", spam_count)
col3.metric("Not Spam", total - spam_count)

st.divider()

# Email input
email_text = st.text_area(
    "Enter your email message below:",
    height=150,
    placeholder="Type or paste your email message here..."
)

if st.button("🔍 Check Email", type="primary"):
    if email_text.strip():
        prediction = model.predict([email_text])[0]
        label = str(prediction).strip().lower()

        is_spam = label in ["spam", "1", "true"]
        result = "SPAM" if is_spam else "NOT SPAM"

        if is_spam:
            st.error("🚨 Result: SPAM EMAIL")
            st.write("This message may be unwanted or suspicious.")
        else:
            st.success("✅ Result: NOT SPAM")
            st.write("The model classified this message as a normal email.")

        st.session_state.history.insert(0, {
            "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Email Message": email_text.strip(),
            "Result": result
        })
    else:
        st.warning("Please enter an email message first.")

st.caption(
    "Student project using a small sample dataset. "
    "Predictions may not always be correct."
)

# Email history
st.divider()
st.subheader("📋 Email Detection History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv_data,
        file_name="email_detection_history.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No emails tested yet. Enter an email above to start.")

# About project
st.divider()
st.subheader("ℹ️ About This Project")
st.write(
    "This project uses Machine Learning to classify email messages "
    "as Spam or Not Spam."
)
