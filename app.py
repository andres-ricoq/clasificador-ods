"""
Aplicación interactiva para clasificar texto libre según el
Objetivo de Desarrollo Sostenible (ODS) más relacionado.

Carga el pipeline entrenado (TF-IDF + SVD + Regresión Logística)
guardado desde el notebook como 'modelo_ods.pkl' y lo usa para
predecir sobre texto ingresado por el usuario.

Para ejecutar:
    streamlit run app.py
"""

import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.stem import SnowballStemmer

# ---------------------------------------------------------------------
# Preprocesamiento de texto
#
# IMPORTANTE: esta función debe ser idéntica a la usada en el notebook
# para entrenar el modelo (sección 1.5). El TfidfVectorizer guardado
# dentro de 'modelo_ods.pkl' la referencia como su `preprocessor`, así
# que si no existe aquí con el mismo comportamiento, joblib.load()
# falla al reconstruir el pipeline.
# ---------------------------------------------------------------------

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('spanish'))
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer('spanish')


def preprocess_text(texto):
    tokens = tokenizer.tokenize(str(texto).lower())
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)


# ---------------------------------------------------------------------
# Nombres de los ODS, para mostrar algo más legible que solo el número
# ---------------------------------------------------------------------

NOMBRES_ODS = {
    1: 'Fin de la pobreza',
    2: 'Hambre cero',
    3: 'Salud y bienestar',
    4: 'Educación de calidad',
    5: 'Igualdad de género',
    6: 'Agua limpia y saneamiento',
    7: 'Energía asequible y no contaminante',
    8: 'Trabajo decente y crecimiento económico',
    9: 'Industria, innovación e infraestructura',
    10: 'Reducción de las desigualdades',
    11: 'Ciudades y comunidades sostenibles',
    12: 'Producción y consumo responsables',
    13: 'Acción por el clima',
    14: 'Vida submarina',
    15: 'Vida de ecosistemas terrestres',
    16: 'Paz, justicia e instituciones sólidas',
    17: 'Alianzas para lograr los objetivos',
}


# ---------------------------------------------------------------------
# Carga del modelo (una sola vez, cacheada entre interacciones)
# ---------------------------------------------------------------------

@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_ods.pkl')


modelo = cargar_modelo()

# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------

st.title('Clasificador de textos según los ODS')
st.write(
    'Ingresa un texto en español y el modelo predice a cuál de los '
    'Objetivos de Desarrollo Sostenible está más relacionado.'
)

texto_usuario = st.text_area(
    'Texto a clasificar',
    height=200,
    placeholder='Pega o escribe aquí el texto que quieres clasificar...'
)

if st.button('Clasificar'):
    if not texto_usuario.strip():
        st.warning('Ingresa un texto antes de clasificar.')
    else:
        # El pipeline aplica internamente preprocess_text -> TF-IDF -> SVD
        # -> clasificador, exactamente igual que en el notebook.
        prediccion = modelo.predict([texto_usuario])[0]
        probabilidades = modelo.predict_proba([texto_usuario])[0]
        confianza = probabilidades.max()

        nombre_ods = NOMBRES_ODS.get(prediccion, 'Desconocido')

        st.success(f'ODS predicho: **{prediccion} — {nombre_ods}**')
        st.write(f'Confianza del modelo: **{confianza:.1%}**')

        # Top 3 ODS más probables, para dar más contexto que solo la
        # predicción principal.
        clases = modelo.classes_
        top3_idx = probabilidades.argsort()[::-1][:3]

        st.write('Top 3 ODS más probables:')
        for idx in top3_idx:
            ods_num = clases[idx]
            st.write(
                f'- ODS {ods_num} ({NOMBRES_ODS.get(ods_num, "?")}): '
                f'{probabilidades[idx]:.1%}'
            )
