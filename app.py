import numpy as np
import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

# Download NLTK resources
nltk.download('stopwords')

# Streamlit app title
st.title("Spam Email Detection Dashboard")


# User interaction
st.header("User Interaction")
name = st.text_input("Enter your name:")
if st.button("Submit"):
    st.write(f"Welcome, {name}!")

# Select box for favorite programming language
programming_language = st.selectbox("Choose your favorite programming language:", ["Python", "Java", "JavaScript", "C++"])
st.write(f"You selected: {programming_language}")

# Slider to select a number from 1 to 100
number = st.slider("Select a number between 1 and 100:", 1, 100)
st.write(f"You selected: {number}")

# Checkbox that displays a message when checked
checkbox = st.checkbox("Show a message")
if checkbox:
    st.write("Checkbox is checked!")

# Radio button to select between different options
radio_option = st.radio("Select your level:", ['Beginner', 'Intermediate', 'Advanced'])
st.write(f"You selected: {radio_option}")

# **Custom Heart Slider**
st.subheader("Rate Your Experience:")
heart_rating = st.slider('Rate our service from 1 to 5 hearts:', min_value=1, max_value=5, value=3)

# Custom UI with emojis (optional)
emojis = ['❤️', '❤️', '❤️', '❤️', '❤️']
st.write("Your Rating: ", ''.join(emojis[:heart_rating]))

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file of emails", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Original Data:")
    st.dataframe(df)

    # Preprocessing function
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text

    # Apply preprocessing
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    st.write("Cleaned Data:")
    st.dataframe(df[['text', 'cleaned_text']])

    # Visualization of spam count
    spam_count = df['label'].value_counts()
    st.bar_chart(spam_count)

    # Word Cloud
    spam_words = ' '.join(df[df['label'] == 'spam']['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Machine Learning Model
    X = df['cleaned_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    # Simulated training history for illustration
    training_accuracy = np.random.rand(10).tolist()  # Replace with actual history
    val_accuracy = np.random.rand(10).tolist()  # Replace with actual validation accuracy history

    # Accuracy Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    st.pyplot(plt)

    # User input for prediction
    email_text = st.text_area("Enter an email to classify:")
    if st.button("Classify"):
        email_vectorized = vectorizer.transform([preprocess_text(email_text)])
        prediction = model.predict(email_vectorized)
        st.write(f"The email is classified as: **{prediction[0]}**")