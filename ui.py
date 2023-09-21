import streamlit as st

import requests
import funciones
import openai
from streamlit_lottie import st_lottie
import xgboost as xgb
import pickle
import pandas as pd

import xgboost as xgb
from sklearn import svm

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


################################################################################################################################################################

def main():
    with st.container():
        #st.header(":red[Rubiales vs Jenny Analisis de Sentimiento] :wave:")
        left,rigth = st.columns(2)
        with left:
            lottie_url = "https://lottie.host/9b1d760a-d152-4817-9a7c-d5dce70d0f96/65tWrCArzp.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json,height=400)
            lottie_url = "https://lottie.host/d27c410d-c34e-494c-826b-47d37805e1e1/VkSmAWhA8B.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json,height=400)
        with rigth:
            st.title(":red[YT Sentiment Inspector v1.0!]")
            
            st.write(""":blue[Description:]

YT Sentiment Inspector is a powerful and versatile tool designed to provide you with valuable information about comments on YouTube videos. It allows you to analyze up to five YouTube videos simultaneously and gain a comprehensive understanding of audience sentiment, comment tone, and keyword frequency.

:blue[How It Works:]

:green[Comment Extraction:] Once you input the video IDs, YT Sentiment Inspector automatically extracts comments from YouTube.

:green[WordCloud and Word Frequency:] The application generates a WordCloud to visualize the most frequent words in comments and a bar chart showing the frequency of the most commonly used words in comments.

:green[Sarcasm Analysis:] YT Sentiment Inspector includes a sarcasm analysis that detects sarcastic comments and displays them in a separate bar chart.

:green[Offensive Language Analysis:] An analysis is also performed to detect offensive language in comments.

:green[Sentiment Analysis:] It uses the NLTK SIA library to predict the overall sentiment of comments. Additionally, another model is employed to detect sentiments such as fear, happiness, sadness, etc.""")
            
    st.write("---")
    id_videos = ["ID1","ID2","ID3","ID4","ID5"]
    input = {}
    lista_videos = []
    
    st.sidebar.text("Introduce ID YT ")
    st.sidebar.image("example.png")
    for idd in id_videos:
        value = st.sidebar.text_input(f'Selecciona {idd}', max_chars=20,disabled=False,value = "")
        input[idd] = value
    for idd in id_videos:
            if input[idd] != "":
                lista_videos.append(input[idd]) # metiendo en una lista los ID de videos introducidos
    if st.sidebar.button("Get Comments!"):
        lista = []
        for video in lista_videos:
            lista.append(funciones.get_all_comments(video)) #extrayendo comentarios de cada uno de los ID introducidos 
            
        data = pd.DataFrame(columns=['text', 'likes', 'dislikes'])
        for i,dataa in enumerate(lista):
            data = pd.concat([data,pd.DataFrame(dataa)])  # creo dataframe con todas las listas de comentarios
        
        st.subheader(f":blue[{len(data)} comentarios capturados.] :yellow[Be Patience!]")
        data["text_stemmer"] = data["text"].apply(funciones.cleantext_to_Stemmer_v2) # PORTER STEMMER
        data["sarcasmo"] = data["text_stemmer"].apply(funciones.busca_sarcasmo)# ETIQUETANDO COMENTARIOS sarcasmo
        data["Ofensivo"] = data["text_stemmer"].apply(funciones.busca_toxity)# ETIQUETANDO COMENTARIOS OFENSIVOS
        data["Emotions"] = data["text_stemmer"].apply(funciones.busca_emotions)# ETIQUETANDO Emociones
        #st.write(data.drop(columns = "text_stemmer"), width=1500)
        
        
        st.subheader(":red[Visualizacion de Datos]")
        funciones.tendencias(data)
        funciones.Sentiment_Intentsity_Analist(data)

    
        
    
    
    
        

    






if __name__ == "__main__":
    main()