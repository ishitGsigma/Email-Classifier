import streamlit as st
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords

# -------------------------------
# Download stopwords once
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)  # remove emails
    text = re.sub(r'http\S+', ' ', text)  # remove urls
    text = ''.join([c for c in text if c not in string.punctuation])  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="wide"
)

# -------------------------------
# Load Pickle Model & Vectorizer
# -------------------------------
model = pickle.load(open("logreg_spam_model.pkl", "rb"))
vectorizer = pickle.load(open("logreg_vectorizer.pkl", "rb"))

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.title { text-align: center; font-size: 45px; color: #4A90E2; font-weight: bold; }
.subtitle { text-align: center; font-size: 18px; color: gray; }
.result-spam { background-color: #ff4b4b; padding: 15px; border-radius: 10px; color: white; font-size: 20px; text-align: center; }
.result-ham { background-color: #2ecc71; padding: 15px; border-radius: 10px; color: white; font-size: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📊 About Project")
st.sidebar.write("""
This AI system detects whether an email/message is **Spam or Not Spam**.

Model Used:
- TF-IDF Vectorizer
- Logistic Regression Classifier

Built with:
- Python
- Streamlit
- Machine Learning
""")
st.sidebar.info("Enter a message in the main panel to test the spam detector.")

# -------------------------------
# Title
# -------------------------------
st.markdown('<p class="title">📧 AI Email Spam Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Check whether a message is Spam or Not using Machine Learning</p>',
            unsafe_allow_html=True)
st.write("")
st.write("")

# -------------------------------
# Input Text Area
# -------------------------------
message = st.text_area("✉️ Enter your message here:", height=150)
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🔍 Analyze Message")

# -------------------------------
# Prediction Logic
# -------------------------------
if predict_button:
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        # Clean & vectorize
        cleaned_msg = clean_text(message)
        vector_input = vectorizer.transform([cleaned_msg])

        # Predict
        prediction = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0][1]

        st.write(f"Spam Probability: {prob:.2%}")

        # Display result
        if prediction == 1:
            st.markdown('<div class="result-spam">🚨 This message is SPAM</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-ham">✅ This message is NOT Spam</div>', unsafe_allow_html=True)

# -------------------------------
# Example Messages
# -------------------------------
st.write("")
st.write("")
st.subheader("🧪 Try Example Messages")

col1, col2 = st.columns(2)
with col1:
    st.info("""
    Spam Examples:
    - Congratulations! You won $5000
    - Claim your FREE iPhone now
    - Urgent! Your bank account needs verification
    - Win a free house by clicking this link
    """)
with col2:
    st.success("""
    Normal Messages:
    - Hey are we meeting tomorrow?
    - Please send the project report
    - Let's have lunch at 2 PM
    - Can you review my assignment?
    """)

# -------------------------------
# Footer
# -------------------------------
st.write("")
st.markdown("---")
st.markdown("💡 Built using Machine Learning with Streamlit")