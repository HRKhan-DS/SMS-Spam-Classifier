import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Classify'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam Message")
        st.write("This message has been classified as spam.")
    else:
        st.header("Not Spam Message")
        st.write("This message is not considered spam.")


# Larger gap using multiple <br> tags
st.markdown("<br><br><br>", unsafe_allow_html=True)

# Instructions for SMS Spam Classifier
st.write("Welcome to the SMS Spam Classifier!")
st.write("To determine if a message is spam or not, please enter the SMS text in the input box.")
st.write("Click the 'Classify' button, and the result (spam or not spam) will be displayed above.")

