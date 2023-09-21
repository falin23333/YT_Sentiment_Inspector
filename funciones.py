import streamlit as st
import pickle
import re
import nltk
import requests
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import itertools
from wordcloud import WordCloud
import plotly.express as px
from sklearn import svm
# Descarga recursos de NLTK si es necesario
nltk.download('punkt')
nltk.download('stopwords')
stopwords_english = stopwords.words('english')
#print(stopwords_spanish[:15]) 
porter = PorterStemmer()
nltk.download('sentiwordnet')
nltk.download('vader_lexicon') 
sia = SentimentIntensityAnalyzer()
import plotly.graph_objs as go
import plotly.subplots as sp




with open(f'models/XGBClassifier_offensive.pkl', 'rb') as f:
    XGB_model_offensive = pickle.load(f)
with open(f'models/RandomForest_model_SARCASM.pkl', 'rb') as f:
    RandomForest_model_SARCASM = pickle.load(f)
with open(f'models/LogisticRegression_emotions.pkl', 'rb') as f:
    LogisticRegression_emotions = pickle.load(f)



with open(f'models/vectorizer_offensive.pkl', 'rb') as f:
    vectorizer_offensive = pickle.load(f)
with open(f'models/vectorizer_SARCASM.pkl', 'rb') as f:
    vectorizer_sarcasmo = pickle.load(f)
with open(f'models/vectorizer_emotions.pkl', 'rb') as f:
    vectorizer_emotions = pickle.load(f)

lista = []




def get_all_comments(video_id):
    lista = []

    def get_comments(page_token=None):
        base_url = 'https://www.googleapis.com/youtube/v3/commentThreads'
        with open(f'models/char.pkl', 'rb') as f:
                char = pickle.load(f)   
        params = {
            'key': char,
            'videoId': video_id,
            'part': 'snippet',
            'maxResults': 100,  # Máximo permitido por página
            'pageToken': page_token
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()

            # Procesa los comentarios aquí
            for item in data.get('items', []):
                comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                like_count = item['snippet']['topLevelComment']['snippet'].get('likeCount', 0)
                dislike_count = item['snippet']['topLevelComment']['snippet'].get('dislikeCount', 0)

                lista.append({
                    'text': comment_text,
                    'likes': like_count,
                    'dislikes': dislike_count
                })

                # Procesa las respuestas a los comentarios principales
                replies = item.get('replies')
                if replies:
                    for reply_item in replies['comments']:
                        reply_text = reply_item['snippet']['textDisplay']
                        reply_like_count = reply_item['snippet'].get('likeCount', 0)
                        reply_dislike_count = reply_item['snippet'].get('dislikeCount', 0)

                        lista.append({
                            'text': reply_text,
                            'likes': reply_like_count,
                            'dislikes': reply_dislike_count
                        })

            # Continúa con la siguiente página si existe
            next_page_token = data.get('nextPageToken')
            if next_page_token:
                get_comments(next_page_token)
        else:
            print(f"Error en la solicitud a la API. Código de estado: {response.status_code}")

    get_comments()
    return lista

def busca_sarcasmo(text):
    # Preprocesamiento del nuevo review
    
   
    new_review_vect = vectorizer_sarcasmo.transform([text])

    # Predicción de la puntuación
    predicted_score = RandomForest_model_SARCASM.predict(new_review_vect)

    return predicted_score[0]

def cleantext_to_Stemmer_v2(text):
    # Convertir todo el texto a minúsculas
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove @mentions
    text = re.sub(r'#', '', text)# remove # tag
    text = re.sub(r'RT[\s]+', '', text) # remove the RT
    text = re.sub(r'https?:\/\/\S+', '', text) # remove links
    text = re.sub('(\\\\u([a-z]|[0-9])+)', ' ', text) # remove unicode characters
    # Eliminar caracteres no alfabéticos ni espacios en blanco
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remover caracteres simples
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remover caracteres simples del inicio
    text = re.sub(r'^[a-zA-Z]\s+', ' ', text) 
    # Remover múltiples espacios con uno solo
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Tokenizar el texto
    text = nltk.word_tokenize(text)
    # Remover stopwords
    text = [word for word in text if not word in stopwords_english]
    # Unir las palabras
    text = ' '.join(text)
    # Stemming
    
    text = porter.stem(text)
    
    return text

def busca_toxity(text):
    # Preprocesamiento del nuevo review
    
    vectorizer = TfidfVectorizer()
    # Vectorización del nuevo review
    new_review_vect = vectorizer_offensive.transform([text])

    # Predicción de la puntuación
    predicted_score = XGB_model_offensive.predict(new_review_vect)

    return predicted_score[0]

def busca_emotions(text):
    # Preprocesamiento del nuevo review
    
    vectorizer = TfidfVectorizer()
    # Vectorización del nuevo review
    new_review_vect = vectorizer_emotions.transform([text])

    # Predicción de la puntuación
    predicted_score = LogisticRegression_emotions.predict(new_review_vect)

    return predicted_score[0]


def tendencias(data):
    
            
            
    filtrar = []
    if True:
        filtrar.append("quot")
        filtrar.append("br")
        filtrar.append("si")
        filtrar.append("39")

    todos = []
    for i in range(data["text_stemmer"].shape[0]):
        titular = data.iloc[i].text_stemmer
        titular = nltk.tokenize.RegexpTokenizer("[\w]+").tokenize(titular)
        
        titular = [word for word in titular if word not in stopwords_english]
        titular = [word for word in titular if word not in filtrar]

        todos.append(titular)
    
    comments_rubiales = list(itertools.chain(*todos))
    freq_comments_rubiales = nltk.FreqDist(comments_rubiales)

    df_freq_comments = pd.DataFrame(list(freq_comments_rubiales.items()), columns = ["Word","Frequency"])
    df_freq_comments.sort_values('Frequency',ascending=False, inplace = True)

    df_freq_comments.reset_index(drop = True, inplace=True)

    top_30_words = df_freq_comments.iloc[:30]

    # Crear el gráfico de barras con Plotly
    
    fig = px.bar(top_30_words, x='Word', y='Frequency', color='Frequency', labels={'Word': 'Palabra', 'Frequency': 'Frecuencia'} )
    fig.update_xaxes(tickangle=45)  # Rotar las etiquetas del eje x para mayor legibilidad

        

        
    
    # Crear un diccionario de palabras y sus frecuencias
    word_freq = dict(zip(df_freq_comments['Word'], df_freq_comments['Frequency']))
    
    wc = WordCloud( max_words=7200, width=1600,height = 1000 , stopwords = stopwords_english).generate_from_frequencies(word_freq)
    def plot_cloud(wc):
        # Set figure size
        plt.figure(figsize = (10,6))
        # Display image
        plt.imshow(wc) 
        # No axis details
        plt.axis("off")
        plt.savefig("wordcloud.png", bbox_inches="tight")
      
        st.image("wordcloud.png")
    st.write(":blue[WORDCLOUD ]")
    plot_cloud(wc)
    #st.write("\n\n\n:blue[Frecuencia de palabras más usadas ]")
    st.write(':blue[Frecuencia de palabras más usadas]')
    st.plotly_chart(fig)
def SIA_POLARITY(texto):
    # Calcular el puntaje de sentimiento del texto
    sentimiento = sia.polarity_scores(texto)
    
    # Verificar si el puntaje compuesto es negativo (indicativo de contenido ofensivo)
    if sentimiento['compound'] < 0:
        return sentimiento
    else:
        return sentimiento    

def Sentiment_Intentsity_Analist(data):
    metricas_sia = data["text_stemmer"].apply(SIA_POLARITY)

    df_metricas_sia = {}
    for i,metricas in enumerate(metricas_sia):
        df_metricas_sia[i] = metricas    
        
    df_metricas_sia = pd.DataFrame(df_metricas_sia).T
    df_metricas_sia.reset_index(inplace = True)
    df_metricas_sia.drop(columns = {"index"},inplace = True)

    data = data.merge(df_metricas_sia, how="right", left_index=True, right_index=True).copy()
    data["Sentiment"] = np.where(data["compound"] == 0, "Neutral",  np.where(data["compound"] < 0,"Negativo","Positivo"  ))
    st.write("---")
    # Crear el gráfico de barras con Plotly
    # Crear el gráfico de barras con Plotly y asignar colores directamente
    st.subheader(":red[Análisis de Comentarios nltk Sentiment Intensity Analist] ")
    with st.container():
        right,left = st.columns(2)
        with left:
        
            fig = px.bar(
                data, 
                x='Sentiment', 
                
                color=data['Sentiment'],
                color_discrete_map={
                    'Positivo': 'green',
                    'Negativo': 'red',
                    'Neutral': 'blue'
                },
                width=400 
            )

            # Mostrar el gráfico en Streamlit
            
            st.plotly_chart(fig)
        with right:
            
            st.write(data[["text","Sentiment"]])
            
    st.write("---")
    st.subheader(":red[Analisis comentarios Ofensivos]")
    with st.container():
        right,left = st.columns(2)
        with left:
        
            # Crear un DataFrame para el gráfico
            data_for_chart = data["Ofensivo"].value_counts().reset_index()
            data_for_chart.columns = ["Ofensivo", "Cantidad"]

            # Asignar etiquetas a los valores de la columna "Ofensivo"
            data_for_chart["Ofensivo"] = data_for_chart["Ofensivo"].map({0: "No Ofensivo", 1: "Ofensivo"})

            # Crear el gráfico de barras con Plotly
            fig = px.bar(
                data_for_chart,
                x="Ofensivo",
                y="Cantidad",
                color="Ofensivo",
                labels={"Ofensivo": "Categoría", "Cantidad": "Cantidad"},
                title="Gráfico de Barras de Variable Ofensivo/No Ofensivo",
                width=400
            )

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig)
            
        with right:
            st.write("\n\n:blue[Comentarios Negativos]\n")    
            df_ofensivo = data[data["Ofensivo"] == 1]
            st.write(df_ofensivo[["Ofensivo","text"]])
    st.write("---")
    st.subheader(":red[Analisis Comentarios Sarcasmo]")
    with st.container():
        right,left = st.columns(2)
        with left:
        
            # Crear un DataFrame para el gráfico
            data_for_chart = data["sarcasmo"].value_counts().reset_index()
            data_for_chart.columns = ["sarcasmo", "Cantidad"]

            # Asignar etiquetas a los valores de la columna "Ofensivo"
            data_for_chart["sarcasmo"] = data_for_chart["sarcasmo"].map({0: "No sarcasmo", 1: "sarcasmo"})

            # Crear el gráfico de barras con Plotly
            fig = px.bar(
                data_for_chart,
                x="sarcasmo",
                y="Cantidad",
                color="sarcasmo",
                labels={"sarcasmo": "Categoría", "Cantidad": "Cantidad"},
                title="Gráfico de Barras de Variable sarcasmo/No sarcasmo",
                width=400
            )

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig)
            
        with right:
            st.write("\n\n:blue[Comentarios sarcasmo]\n")    
            df_sarcasmo = data[data["sarcasmo"] == 1]
            st.write(df_sarcasmo[["sarcasmo","text"]])



    st.write("---")
    st.subheader(":red[Analisis Emociones]")
    with st.container():
        right,left = st.columns(2)
        with left:
            
           # Crear un DataFrame con la distribución de emociones
            emotions_df = data['Emotions'].value_counts().reset_index()
            emotions_df.columns = ['Emotions', 'Count']

            # Crear el gráfico de barras con Plotly
            fig = px.bar(emotions_df, x='Emotions', y='Count', labels={'Emotions': 'Emotions Label', 'Count': 'Count'},width=400)
            fig.update_layout(title='Distribución de Emociones')
            fig.update_xaxes(type='category')

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig)

           
            
        with right:
            st.write("\n\n:blue[]\n")    
            #df_ofensivo = data[data["sarcasmo"] == 1]
            st.write(data[["Emotions","text"]])