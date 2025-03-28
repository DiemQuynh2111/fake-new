import streamlit as st
import pickle
import re
import string
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

vector_form = pickle.load(open('vector.pkl', 'rb'))  
load_model = pickle.load(open('model.pkl', 'rb'))  

stop_words = set(stopwords.words('english'))

def xuli(input_text):
    input_text = input_text.lower()
    input_text = re.sub(f"[{re.escape(string.punctuation)}]", '', input_text)
    words = input_text.split()
    filtered_text = ' '.join([word for word in words if word not in stop_words])
    return filtered_text

def fake_news(news):
    news = xuli(news)
    input_data = [news]
    
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except:
        return None


st.title('Fake News Classification App')
st.subheader("Choose to input news content or a URL")

input_option = st.radio("Select Input Method:", ("Text", "URL"))

if input_option == "Text":
    sentence = st.text_area("Enter your news content here", "", height=200)
elif input_option == "URL":
    url = st.text_input("Enter the news article URL here")
    if url:
        sentence = get_text_from_url(url)
        if sentence:
            st.write("Extracted Text:", sentence)
        else:
            st.error("Could not retrieve text from URL.")

if st.button("Predict"):
    if sentence.strip():
        prediction_class = fake_news(sentence)
        st.write(f'Prediction: {prediction_class}')

        if prediction_class[0].lower() == 'real':
            st.success('Prediction: Reliable')
        elif prediction_class[0].lower() == 'fake':
            st.warning('Prediction: Unreliable')
        else:
            st.error('Unexpected prediction output. Please check the model.')
    else:
        st.warning("Please enter some text or URL to classify.")