import streamlit as st
import numpy as np
import pandas as pd
import pickle


with open("SVCModel.pkl", "rb") as model_file:
    svc_model = pickle.load(model_file)

df = pd.read_csv('new_data.csv')

def main():
    st.title("Fake SMS Classification")

    # Create an input text area for the user to enter news content
    input_data = st.text_area("Enter the SMS text:")

    if st.button("Classify"):
        # Make sure the input data is not empty
        if input_data:
            # Perform any preprocessing on input_data if necessary (e.g., vectorization)
            # Here, you should apply the same preprocessing steps used when training the model

            # Predict the class of the news
            prediction = svc_model.predict([input_data])

            # Display the prediction result
            if prediction[0] == 0:
                st.write('The SMS is classified as Real')
            else:
                st.write('The SMS is classified as Fake')
        else:
            st.write("Please enter some SMS text before classifying.")

    # Larger gap using multiple <br> tags
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # Instructions for SMS Spam Classifier
    st.write("Welcome to the Fake SMS Classifier!")
    st.write("To determine whether the SMS is true or fake, enter some text into the box.")
    st.write("Click the 'Classify' button, and the result (Real or fake) will be displayed above.")

if __name__ == '__main__':
    main()

