import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
import sys
import traceback
from sklearn.preprocessing import LabelEncoder as le
import geopandas as gpd
import matplotlib.pyplot as plt



def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl('https://lottie.host/5d31c170-a371-43bf-8bc8-f177330eb9fa/e8QfUTpJuJ.json')
lottie_coding_2 = load_lottieurl("https://lottie.host/97516d4f-70e1-4191-8d2b-e8553fe6f4af/gMaiFE4D9i.json")
lottie_coding_3 = load_lottieurl('https://lottie.host/3c5475a5-8a8a-42f1-8cd5-74cb4198395c/p2c0DjiSL6.json')
lottie_coding_4 = load_lottieurl('https://lottie.host/6c062e98-f431-4adb-a677-38e536a7f10a/t21eGU3B1G.json')
imagen = Image.open('/content/sample_data/CLASIFICACION.png')

with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
      st.subheader('Proyecto de Analítica Aplicada - Universidad de La Sabana')
      st.title('**Bienvenido a la plataforma de Escudo Turístico** :wave:')

    with right_column:
      st_lottie(lottie_coding, height=300, key="code")


with st.container():
    st.write("---")
    st.header('Objetivo')
    st.write('Ayudar a los turistas locales y extranjeros a identificar accesiblemente el nivel de inseguridad en diferentes zonas del país para contar con más información al planear su viaje')
    left_column, right_column = st.columns(2)
    with left_column:
        st.write('Creadoras:')
        st.markdown("""
            >- *María Alejandra Castañeda Sánchez*
            >- *Ana Conti*
            >- *María Paula D'asté*
            >- *Danna Durán*
        """)

    with right_column:
        st_lottie(lottie_coding_2, height=200, key="coding")

with st.container():
    st.write("---")
    st.header('¿Cómo funciona el Escudo Turístico?')
    left_column, right_column = st.columns(2)
    with left_column:
        st.write('Para definir el nivel de inseguridad en cada departamento se creó una clasificación basada en los ‘Travel Advisories’ que emite The Department of State’s Bureau of Consular Affairs para los ciudadanos estadounidenses')
        st.write('Se evalúa cada departamento según los siguientes parámetros:')
    with right_column:
        st_lottie(lottie_coding_3, height=200, key="codi")

with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
      st_lottie(lottie_coding_4, height=300, key="cod")

    with right_column:
      st.markdown("""
            >- *Presencia de grupos armados*
            >- *Número de homicidios*
            >- *Número de delitos sexuales presentados*
            >- *Número de lesiones personales*
            >- *Número de secuestros*
            >- *Número de actos terroristas*
            >- *Otros delitos (hurto a vivienda, comercios, etc...)*
            >- *Presencia de desastres naturales*
            >- *Presencia de estaciones de policía*
            >- *Presencia de centros de salud*
        """)

with st.container():
    st.write("---")
    st.header('Clasificación actual de los departamentos de Colombia')
    st.write('Según la información actual, el nivel de inseguridad por cada departamento se muestra a continuación aplicado al contexto de recomendación de viajes')
    st.image(imagen)

ponderacion = {
    'homicidio': 0.6,
    'del_sexual': 0.5,
    'les_personal': 0.5,
    'grupos_armados': 50,
    'secuestro': 0.5,
    'terrorismo': 0.5,
    'otros_delitos': 0.4,
    'desastres_naturales': 0.1,
    'estaciones_policia': -0.2,
    'centros_salud': -0.2
}

with st.container():
    st.write("---")
    st.header('Clasificación del departamento al que deseas viajar')
    st.write('Introduce la información requerida para conocer la clasificación del departamento según la situación dada:')
    st.write('A continuación se muestra el código para cada departamento:')
    left_column, right_column = st.columns(2)
    with left_column:

      departamentos = {
        'Amazonas': 1, 'Antioquia': 2, 'Arauca': 3,
        'Archipiélago de San Andrés, Providencia y Santa Catalina': 4, 'Atlántico': 5, 'Bolívar': 6,
        'Boyacá': 7, 'Caldas': 8, 'Caquetá': 9,
        'Casanare': 10, 'Cauca': 11, 'Cesar': 12,
        'Chocó': 13, 'Córdoba': 14, 'Cundinamarca': 15,
        'Guainía': 16}

      for departamento, codigo in departamentos.items():
        st.write(f"{departamento} = {codigo}")

    with right_column:

      departamentos = {'Guaviare': 17, 'Huila': 18,
        'La Guajira': 19, 'Magdalena': 20, 'Meta': 21,
        'Nariño': 22, 'Norte de Santander': 23, 'Putumayo': 24,
        'Quindío': 25, 'Risaralda': 26, 'Santander': 27,
        'Sucre': 28, 'Tolima': 29, 'Valle del Cauca': 30,
        'Vaupés': 31, 'Vichada': 32}

      for departamento, codigo in departamentos.items():
          st.write(f"{departamento} = {codigo}")

with st.container():

    departamento = st.number_input('Indica el código del departamento:', min_value = 1, step=1, max_value=32)

    año_seleccionado = st.number_input('Indica el año:', min_value = 2019, step=1, max_value=2023)

    mes_seleccionado = st.number_input('Indica el mes en términos númericos:', min_value=1, step=1, max_value=12)

    otros_delitos = st.number_input('Indica la cantidad de otros delitos:', min_value=0, step=1)

    secuestro = st.number_input('Indica la cantidad de secuestros:', min_value=0, step=1)

    terrorismo = st.number_input('Indica la cantidad de actos de terrorismo:', min_value=0, step=1)

    del_sexual = st.number_input('Indica la cantidad de delitos sexuales:', min_value=0, step=1)

    homicidio = st.number_input('Indica la cantidad de homicidios:', min_value=0, step=1)

    les_personal = st.number_input('Indica la cantidad de lesiones personales:', min_value=0, step=1)

    estaciones_policia = st.number_input('Indica la cantidad de estaciones de policía:', min_value=0, step=1)

    centros_salud = st.number_input('Indica la cantidad de centros de salud:', min_value=0, step=1)

    grupos_armados = st.number_input('Indica el nivel de presencia de grupos armados:', min_value=0, step=5, max_value=10)

    desastres_naturales = st.number_input('Indica el nivel de probabilidad de desastres naturales:', min_value=1, step=1, max_value= 5)


    puntaje_total = ( otros_delitos * ponderacion['otros_delitos'] + secuestro * ponderacion['secuestro'] + terrorismo * ponderacion['terrorismo'] + del_sexual * ponderacion['del_sexual'] +
    homicidio * ponderacion['homicidio'] + les_personal * ponderacion['les_personal'] + estaciones_policia * ponderacion['estaciones_policia'] +
    centros_salud * ponderacion['centros_salud'] + grupos_armados * ponderacion['grupos_armados'] + desastres_naturales * ponderacion['desastres_naturales']
)

if st.button("Clasificación:"):

  try:

    model = joblib.load('/content/modelo_knn.joblib')

    input_data = np.array([[departamento, año_seleccionado, mes_seleccionado, otros_delitos, secuestro,
                            terrorismo, del_sexual, homicidio, les_personal, estaciones_policia,
                            centros_salud, grupos_armados, desastres_naturales, puntaje_total]])

    prediction = model.predict(input_data)

    if prediction == 0:

      etiqueta_riesgo_predicha = 'Level 2 - Exercise Increased Caution'

    elif prediction == 1:

      etiqueta_riesgo_predicha = 'Level 4 – Do Not Travel'

    elif prediction == 2:

      etiqueta_riesgo_predicha = 'Level 3 - Reconsider Travel'

    elif prediction == 3:

      etiqueta_riesgo_predicha = 'Level 1 - Exercise Normal Precautions'


    st.subheader("¿Qué nivel de inseguridad tiene este departamento?")
    st.write(etiqueta_riesgo_predicha)

    geojson_file = '/content/sample_data/colombia.geo.json'
    colombia = gpd.read_file(geojson_file)

    color_dict = {
    'Level 1 - Exercise Normal Precautions': 'green',
    'Level 2 - Exercise Increased Caution': 'yellow',
    'Level 3 - Reconsider Travel': 'orange',
    'Level 4 – Do Not Travel': 'red'
}


    departamentos = {
    'AMAZONAS': 1,
    'ANTIOQUIA': 2,
    'ARAUCA': 3,
    'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA': 4,
    'ATLANTICO': 5,
    'BOLIVAR': 6,
    'BOYACA': 7,
    'CALDAS': 8,
    'CAQUETA': 9,
    'CASANARE': 10,
    'CAUCA': 11,
    'CESAR': 12,
    'CHOCO': 13,
    'CORDOBA': 14,
    'CUNDINAMARCA': 15,
    'GUAINIA': 16,
    'GUAVIARE': 17,
    'HUILA': 18,
    'LA GUAJIRA': 19,
    'MAGDALENA': 20,
    'META': 21,
    'NARIÑO': 22,
    'NORTE DE SANTANDER': 23,
    'PUTUMAYO': 24,
    'QUINDIO': 25,
    'RISARALDA': 26,
    'SANTANDER': 27,
    'SUCRE': 28,
    'TOLIMA': 29,
    'VALLE DEL CAUCA': 30,
    'VAUPES': 31,
    'VICHADA': 32,
    'SANTAFE DE BOGOTA D.C':33
}

    departamento_nombre = [k for k, v in departamentos.items() if v == departamento][0]

        # Asignar etiqueta de riesgo predicha solo al departamento seleccionado
    colombia['clasificacion'] = np.where(colombia['NOMBRE_DPT'] == departamento_nombre, etiqueta_riesgo_predicha, '')

        # Asignar color al departamento seleccionado según 'clasificacion'
    colombia['color'] = colombia['clasificacion'].map(color_dict).fillna('gray')

        # Crear el mapa
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colombia.plot(color=colombia['color'], ax=ax, legend=False)

        # Guardar la imagen del mapa
    plt.savefig('mapa_colombia.png')

        # Mostrar la imagen en Streamlit
    st.image('mapa_colombia.png', use_column_width=True)

  except Exception as e:

    st.error(f"No puede realizarse la clasificación. Los datos no son válidos, por favor ingreselos nuevamente. Error: {e}")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Error:", exc_value)
    print("Traceback:")
    traceback.print_tb(exc_traceback)
