import streamlit as st
import pandas as pd
import re

# =====================================================
# 🧹 CLEANING FUNCTION
# =====================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text


# =====================================================
# 📦 LOAD DATA + TRAIN MODEL (ONE TIME)
# =====================================================
@st.cache_resource
def load_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # dataset
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None)
    df.columns = ['label', 'message']

    # cleaning
    df['message'] = df['message'].apply(clean_text)

    # vectorization
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # models
    model_nb = MultinomialNB().fit(X_train, y_train)
    model_lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    model_svm = SVC().fit(X_train, y_train)

    # accuracy
    from sklearn.metrics import accuracy_score
    acc_nb = accuracy_score(y_test, model_nb.predict(X_test))
    acc_lr = accuracy_score(y_test, model_lr.predict(X_test))
    acc_svm = accuracy_score(y_test, model_svm.predict(X_test))

    # best model
    best_model = max(
        [(model_nb, acc_nb, "Naive Bayes"),
         (model_lr, acc_lr, "Logistic Regression"),
         (model_svm, acc_svm, "SVM")],
        key=lambda x: x[1]
    )

    return vectorizer, best_model


# =====================================================
# 🎨 UI START
# =====================================================
st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 SMS Spam Classifier")
st.write("Enter a message and check if it's **Spam or Ham**")

# load model
vectorizer, (model, acc, model_name) = load_model()

# show model info
st.success(f"Best Model: {model_name} (Accuracy: {acc:.2f})")

# input box
user_input = st.text_area("✉️ Enter your message here:")

# button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "spam":
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is HAM (Not Spam)")