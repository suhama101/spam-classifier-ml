# =====================================================
# localsystem datset1
# import pandas as pd
# df = pd.read_csv("sms.tsv", sep='\t', header=None)
# df.columns = ['label', 'message']
# print(df.head())
# online link
# 📦 DATASET 1 (BEST - UCI SMS SPAM)
# =====================================================
# url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
# df = pd.read_csv(url, sep='\t', header=None)
# df.columns = ['label', 'message']
# =====================================================
# 📦 DATASET 2 (KAGGLE MIRROR - WORKING)
# 👉 uncomment to use this
# =====================================================
# url = "https://raw.githubusercontent.com/ifrankandrade/sms-spam-collection/master/spam.csv"
# df = pd.read_csv(url, encoding='latin-1')
# df = df[['v1', 'v2']]
# df.columns = ['label', 'message']
# =====================================================
# 📦 DATASET 3 (SMALL TEST DATASET)
# =====================================================
# url = "https://raw.githubusercontent.com/plotly/datasets/master/sms_spam.csv"
# df = pd.read_csv(url)
# df.columns = ['label', 'message']


import pandas as pd
import re

# =====================================================
# 📦 DATASET
# =====================================================
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None)
df.columns = ['label', 'message']


# =====================================================
# 🧹 CLEANING
# =====================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['message'] = df['message'].apply(clean_text)


# =====================================================
# 🔤 TF-IDF
# =====================================================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    min_df=2
)

# 🔥 FIX HERE (THIS WAS MISSING)
X = vectorizer.fit_transform(df['message'])


# =====================================================
# 📊 SPLIT
# =====================================================
from sklearn.model_selection import train_test_split

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================================
# 🤖 MODEL 1: NAIVE BAYES
# =====================================================
from sklearn.naive_bayes import MultinomialNB

model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
pred_nb = model_nb.predict(X_test)


# =====================================================
# 🤖 MODEL 2: LOGISTIC REGRESSION
# =====================================================
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)


# =====================================================
# 🤖 MODEL 3: SVM (PRO LEVEL ADDITION)
# =====================================================
from sklearn.svm import LinearSVC

model_svm = LinearSVC()
model_svm.fit(X_train, y_train)
pred_svm = model_svm.predict(X_test)


# =====================================================
# 📊 EVALUATION
# =====================================================
from sklearn.metrics import accuracy_score

print("\n==============================")
print("📊 MODEL COMPARISON")
print("==============================")

acc_nb = accuracy_score(y_test, pred_nb)
acc_lr = accuracy_score(y_test, pred_lr)
acc_svm = accuracy_score(y_test, pred_svm)

print("\n🔵 Naive Bayes Accuracy:", acc_nb)
print("🟢 Logistic Regression Accuracy:", acc_lr)
print("🟣 SVM Accuracy:", acc_svm)


# =====================================================
# 🏆 BEST MODEL AUTO SELECT
# =====================================================
models = {
    "Naive Bayes": model_nb,
    "Logistic Regression": model_lr,
    "SVM": model_svm
}

best_model_name = max(
    zip(models.keys(), [acc_nb, acc_lr, acc_svm]),
    key=lambda x: x[1]
)[0]

best_model = models[best_model_name]

print("\n🏆 Best Model Selected:", best_model_name)


# =====================================================
# 🧪 LIVE TEST
# =====================================================
while True:
    user_input = input("\nEnter message (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    test_vector = vectorizer.transform([user_input])
    result = best_model.predict(test_vector)

    print("Prediction:", result[0])