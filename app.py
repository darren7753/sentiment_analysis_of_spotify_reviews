# Importing required libraries
import re
import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from streamlit_option_menu import option_menu
from plotly.subplots import make_subplots
from joblib import load

# Downloading required NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Setting page configuration for Streamlit app
st.set_page_config(
    page_title="Sentimen Ulasan Spotify",
    layout="wide"
)

# Function to label the score as 'Negatif' or 'Positif'
def labeling(score):
    if score < 4:
        return "Negatif"
    elif score > 3 :
        return "Positif"

# Function to fetch data from CSV file and preprocess it
@st.cache_data
def fetch_data(file):
    data = pd.read_csv(file)
    data.columns = ["content", "score"]
    data["label"] = data["score"].apply(labeling)
    return data

# Function to convert text to lowercase
def casefolding(text):
    text = text.lower()
    return text

# Function to clean text by removing URLs, mentions, hashtags, numbers, and punctuations
def cleaning(text):
    text = re.sub(r"@[A-Za-a0-9]+", " ",text)
    text = re.sub(r"#[A-Za-z0-9]+", " ",text)
    text = re.sub(r"http\S+", " ",text)
    text = re.sub(r"[0-9]+", " ",text)
    text = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", text)
    text = text.strip(" ")
    return text

# Function to remove emojis from text
def emoji(text):
    return text.encode("ascii", "ignore").decode("ascii")

# Function to replace repeated characters with a single character
def replace(text):
    pola = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pola.sub(r"\1", text)

# Function to tokenize text into words
def tokenizing(text):
    text = word_tokenize(text)
    return text

# List of Indonesian stopwords
daftar_stopword_id = stopwords.words("indonesian")
# Function to remove Indonesian stopwords from text
def stopwordText_ID(text):
    return [word for word in text if word not in daftar_stopword_id]

# List of English stopwords
daftar_stopwor_en = stopwords.words("english")
# Function to remove English stopwords from text
def stopwordText_EN(text):
    return [word for word in text if word not in daftar_stopwor_en]

# Creating an instance of Indonesian stemmer
stemmer_id = StemmerFactory().create_stemmer()
# Function to stem Indonesian words
def stemmed_wrapper_ID(text):
    return stemmer_id.stem(text)

# Creating an instance of English stemmer
stemmer_en = PorterStemmer()
# Function to stem English words
def stemmed_wrapper_EN(term):
    return stemmer_en.stem(term)

# Function to join a list of words into a single string
def join_text_list(text):
    return " ".join(text)

# Function to preprocess text based on the selected language
def preprocess_text(text, language):
    text = casefolding(text)
    text = cleaning(text)
    text = emoji(text)
    text = replace(text)
    text = tokenizing(text)
    
    if language == "🇮🇩 ID":
        text = stopwordText_ID(text)
        text = [stemmed_wrapper_ID(term) for term in text]
    else:
        text = stopwordText_EN(text)
        text = [stemmed_wrapper_EN(term) for term in text]

    return text

# Function to display the home page
def home_page():
    df_ulasan_id = fetch_data("Ulasan Spotify id.csv")
    df_ulasan_en = fetch_data("Ulasan Spotify En.csv")

    st.markdown("<h1 style='text-align: center;'>Aplikasi Prediksi Sentimen Ulasan Spotify</h1>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Displaying welcome message and application description
    st.markdown(
        "<p style='font-size:17px;'>Selamat datang di prediksi ulasan aplikasi Spotify dengan menggunakan algoritma Random Forest. Aplikasi ini dapat digunakan untuk melakukan prediksi ulasan aplikasi Spotify dengan kategori positif dan negatif berdasarkan teks yang diinput.</p>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    # Displaying Spotify logo
    with open("Spotify Logo.svg", "r") as f:
        spotify_logo = f.read()
    st.markdown(f"<div style='text-align:center;'>{spotify_logo}</div>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Displaying information about Spotify application
    st.markdown(
        "<p style='font-size:17px;'>Spotify merupakan aplikasi streaming musik digital yang banyak digunakan untuk mengakses lagu, podcast, dan konten audio dari seluruh dunia. Aplikasi ini termasuk aplikasi yang banyak diunduh dan mendapat penilaian rata-rata 4.4, menjadi salah satu aplikasi streaming musik yang banyak digunakan dan populer di dunia serta bisa diakses secara premium atau pun gratis.</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size:17px;'>Aplikasi ini memungkinkan pengguna untuk mendengarkan jutaan lagu dari berbagai genre musik secara online. Pengguna dapat membuat daftar putar pribadi, menelusuri berbagai artis dan album serta menemukan lagu atau konten audio berdasarkan riwayat lagu-lagu yang telah didengarkan dan disukai sebelumnya.</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size:17px;'>Penelitian dilakukan dengan jumlah keseluruhan 2000 ulasan yang dikumpulkan, terdiri dari 1000 data ulasan bahasa Inggris dan 1000 data ulasan Bahasa Indonesia. Berikut jumlah data dari masing-masing ulasan berdasarkan rating:</p>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    # Displaying bar charts for data counts
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Diagram Jumlah Data Ulasan Bahasa Indonesia",
            "Diagram Jumlah Data Ulasan Bahasa Inggris"
        )
    )

    score_counts_id = df_ulasan_id["score"].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=score_counts_id.index, 
            y=score_counts_id.values, 
            text=score_counts_id.values, 
            textposition="auto", 
            textfont=dict(size=20),
            hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>"
        ),
        row=1, col=1
    )

    score_counts_en = df_ulasan_en["score"].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=score_counts_en.index, 
            y=score_counts_en.values, 
            text=score_counts_en.values, 
            textposition="auto", 
            textfont=dict(size=20),
            hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>"
        ),
        row=1, col=2
    )

    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=25), height=400)
    fig.update_xaxes(title_text="Score", showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="Count", showgrid=False, row=1, col=1)
    fig.update_xaxes(title_text="Score", showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text="Count", showgrid=False, row=1, col=2)

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.write("")

    st.markdown(
        "<p style='font-size:17px;'>Dalam penelitian ini labeling dilakukan dengan rating dengan ketentuan rating 1, 2, dan 3 adalah negatif, sedangkan 4 dan 5 adalah positif. Maka hasil presentase labeling positif dan negatif dapat dilihat pada gambar berikut:</p>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    # Displaying pie charts for positive and negative review percentages
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Persentase Data Ulasan Bahasa Indonesia Positif dan Negatif",
            "Persentase Data Ulasan Bahasa Inggris Positif dan Negatif"
        ),
        specs=[[{"type": "pie"}, {"type": "pie"}]],
    )

    label_counts_id = df_ulasan_id["label"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=label_counts_id.index, 
            values=label_counts_id.values, 
            textinfo="percent+label", 
            textfont=dict(size=15),
            name="Bahasa Indonesia"
        ),
        row=1, col=1
    )

    label_counts_en = df_ulasan_en["label"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=label_counts_en.index, 
            values=label_counts_en.values, 
            textinfo="percent+label", 
            textfont=dict(size=15),
            name="Bahasa Inggris"
        ),
        row=1, col=2
    )

    fig.update_layout(
        showlegend=False, 
        margin=dict(b=30, t=70),
        height=400,
        annotations=[
            dict(
                text="Persentase Data Ulasan Bahasa Indonesia Positif dan Negatif",
                y=1.05, font_size=15, showarrow=False
            ),
            dict(
                text="Persentase Data Ulasan Bahasa Inggris Positif dan Negatif",
                y=1.05, font_size=15, showarrow=False
            )
        ]
    )
    fig.update_traces(hole=.4)

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.write("")

    st.markdown(
        "<p style='font-size:17px;'>Jumlah data yang telah diberikan label pada ulasan bahasa Indonesia, untuk ulasan positif sebanyak 32.3% atau 323. Sedangkan, untuk ulasan negatif sebanyak 67.7% atau 677. Jumlah data yang telah diberikan label pada ulasan bahasa Inggris, untuk ulasan positif sebanyak 13.3% atau 109. Sedangkan, untuk ulasan negatif sebanyak 89.1% atau 891.</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size:17px;'>Dari persentase tersebut dapat disimpulkan bahwa pengguna aplikasi Spotify merasa tidak puas terhadap aplikasi, sehingga perlu adanya perbaikan atau evaluasi bagi pihak terkait untuk meningkatkan atau memperbaiki layanan pada aplikasinya.</p>",
        unsafe_allow_html=True
    )

    # Displaying information about oversampling
    st.divider()
    st.subheader("Oversampling")

    st.markdown(
        "<p style='font-size:17px;'>Tahapan ini dilakukan karena pada jumlah data hasil labeling sebelumnya terdapat perbedaan atau tidak seimbang antara label positif dan negatif. Data yang telah melewati tahap pelabelan akan diketahui jumlah dari label positif dan negatif, dalam penelitian ini terdapat ketidakseimbangan hasil. Karena hal tersebut, maka dilakukan random oversampling. Setelah proses oversampling dilakukan, hasil dari proses tersebut ditampilkan pada gambar berikut:</p>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    before_oversampling_id = pd.Series({"Negatif": 677, "Positif": 323})
    after_oversampling_id = pd.Series({"Negatif": 677, "Positif": 677})

    before_oversampling_en = pd.Series({"Negatif": 891, "Positif": 109})
    after_oversampling_en = pd.Series({"Negatif": 891, "Positif": 891})

    # Displaying bar charts for data counts before and after oversampling (Indonesian)
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=(
            "Jumlah Data Bahasa Indonesia Sebelum Oversampling", 
            "Jumlah Data Bahasa Indonesia Sebelum Oversampling"
        )
    )

    fig.add_trace(
        go.Bar(
            x=before_oversampling_id.index,
            y=before_oversampling_id.values,
            name="Sebelum Oversampling",
            text=before_oversampling_id.values,
            textposition="auto",
            textfont=dict(size=20),
            hovertemplate="Label: %{x}<br>Jumlah Data: %{y}<extra></extra>"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=after_oversampling_id.index,
            y=after_oversampling_id.values,
            name="Sesudah Oversampling",
            text=after_oversampling_id.values,
            textposition="auto",
            textfont=dict(size=20),
            hovertemplate="Label: %{x}<br>Jumlah Data: %{y}<extra></extra>"
        ),
        row=1, col=2
    )

    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=25), height=400)
    fig.update_xaxes(title_text="Label", showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="Jumlah Data", showgrid=False, row=1, col=1)
    fig.update_xaxes(title_text="Label", showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text="Jumlah Data", showgrid=False, row=1, col=2)

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.write("")

    st.markdown(
        "<p style='font-size:17px;'>Data ulasan bahasa Indonesia sebelum Oversampling berjumlah 1000 data, dengan jumlah ulasan positif  323 dan negatif  677.  Setelah proses Oversampling berjumlah 1354 data, dengan jumlah ulasan positif 677 dan negatif 677.</p>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    # Displaying bar charts for data counts before and after oversampling (English)
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=(
            "Jumlah Data Bahasa Inggris Sebelum Oversampling", 
            "Jumlah Data Bahasa Inggris Sebelum Oversampling"
        )
    )

    fig.add_trace(
        go.Bar(
            x=before_oversampling_en.index,
            y=before_oversampling_en.values,
            name="Sebelum Oversampling",
            text=before_oversampling_en.values,
            textposition="auto",
            textfont=dict(size=20),
            hovertemplate="Label: %{x}<br>Jumlah Data: %{y}<extra></extra>"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=after_oversampling_en.index,
            y=after_oversampling_en.values,
            name="Sesudah Oversampling",
            text=after_oversampling_en.values,
            textposition="auto",
            textfont=dict(size=20),
            hovertemplate="Label: %{x}<br>Jumlah Data: %{y}<extra></extra>"
        ),
        row=1, col=2
    )

    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=25), height=400)
    fig.update_xaxes(title_text="Label", showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="Jumlah Data", showgrid=False, row=1, col=1)
    fig.update_xaxes(title_text="Label", showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text="Jumlah Data", showgrid=False, row=1, col=2)

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.write("")

    st.markdown(
        "<p style='font-size:17px;'>Data ulasan bahasa Inggris sebelum Oversampling berjumlah 1000 data, dengan jumlah ulasan positif  109 dan negatif 891.  Setelah proses Oversampling berjumlah 1782 data, dengan jumlah ulasan positif 891 dan negatif 891.</p>",
        unsafe_allow_html=True
    )

# Function to display the prediction page
def prediksi_page():
    st.markdown("<h1 style='text-align: center;'>Prediksi Ulasan Spotify</h1>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    result = None
    error_message = None

    # Displaying input fields for language and text review
    col1, col2, col3 = st.columns([0.3, 1, 0.1])
    with col1:
        language = st.selectbox(
            "Bahasa",
            ["🇮🇩 ID", "🇺🇸 EN"],
            index=None,
            label_visibility="collapsed",
            placeholder="Pilih bahasa"
        )

    with col2:
        text_input = st.text_input("Teks Ulasan", placeholder="Masukkan teks ulasan", label_visibility="collapsed")

    with col3:
        pred_button = st.button("Prediksi", type="primary", use_container_width=True)

        # Checking if the prediction button is clicked
        if pred_button:
            # Validating input fields
            if not language or language == "":
                if not text_input:
                    error_message = "Pilih bahasa dan masukkan teks ulasan terlebih dahulu"
                else:
                    error_message = "Pilih bahasa terlebih dahulu"
            elif not text_input:
                error_message = "Masukkan teks ulasan terlebih dahulu"
            else:
                # Loading the appropriate model and vectorizer based on the selected language
                if language == "🇮🇩 ID":
                    model = load("rf_model_id.joblib")
                    vectorizer = load("vectorizer_id.joblib")
                elif language == "🇺🇸 EN":
                    model = load("rf_model_en.joblib")
                    vectorizer = load("vectorizer_en.joblib")
                
                # Preprocessing the input text
                preprocessed_text = preprocess_text(text_input, language)
                text_input = join_text_list(preprocessed_text)

                # Vectorizing the preprocessed text
                vectorized_text_input = vectorizer.transform([text_input])

                # Making the prediction
                result = model.predict(vectorized_text_input)[0]
                probability = max(model.predict_proba(vectorized_text_input)[0])

    # Displaying the result
    if error_message:
        st.error(error_message, icon="🚨")
    elif result is not None:
        st.info(f"Ulasan tersebut {result.upper()} dengan probabilitas {(probability * 100):.2f}%", icon="ℹ️")

# Displaying the navigation menu
page_options = option_menu(
    menu_title=None,
    options=["Home", "Prediksi"],
    icons=["house-fill", "bar-chart-fill"],
    orientation="horizontal",
    default_index=0
)

# Rendering the selected page
if page_options == "Home":
    home_page()
else:
    prediksi_page()