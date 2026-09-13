# Recomendador de Netflix

En este proyecto he desarrollado un sistema capaz de recomendar películas y series similares utilizando Python, scikit-learn y Streamlit.

Mi objetivo ha sido recorrer el proceso completo de un proyecto de datos: explorar un dataset real, limpiarlo, preparar sus variables, construir el recomendador, comprobar su funcionamiento con tests y crear una aplicación web interactiva.



## ¿Qué hace el proyecto?

La aplicación permite seleccionar una película o serie del catálogo y obtener una lista de títulos similares.

Para generar las recomendaciones tiene en cuenta:

- Los géneros.
- El director.
- El reparto.
- El país de producción.
- La descripción del contenido.

El catálogo utilizado contiene 5.967 películas y series.

## ¿Cómo funciona?

He construido un recomendador basado en contenido. Esto significa que compara las características de los títulos, sin utilizar datos personales ni historiales de visualización.

El proceso que he seguido es el siguiente:

1. Cargo el catálogo desde un archivo CSV.
2. Compruebo que contiene las columnas necesarias.
3. Limpio los valores ausentes y convierto los tipos de datos.
4. Normalizo los géneros, directores, actores y países.
5. Combino las características relevantes en una sola columna de texto.
6. Transformo ese texto en vectores numéricos mediante TF-IDF.
7. Calculo la similitud del coseno entre los títulos.
8. Devuelvo los contenidos con mayor similitud.

Para dar más importancia a determinadas características, he utilizado estos pesos:

- Géneros: peso 3.
- Director: peso 2.
- Reparto: peso 1.
- País de producción: peso 1.
- Descripción: peso 1.

La puntuación mostrada por la aplicación representa la similitud entre las características de los títulos. No es una probabilidad de que una película o serie vaya a gustarle al usuario.

## Funcionalidades

- Búsqueda entre 5.967 títulos.
- Selección mediante título, año y tipo de contenido.
- Recomendaciones basadas en las características del catálogo.
- Elección del número de recomendaciones.
- Visualización de la puntuación IMDb.
- Porcentaje de similitud para cada resultado.
- Interfaz web desarrollada con Streamlit.
- Control de errores y validación de los datos.
- Tests automatizados con Pytest.

## Tecnologías utilizadas

- Python 3.12
- pandas
- NumPy
- SciPy
- scikit-learn
- Streamlit
- Pytest
- Ruff
- Jupyter Notebook

## Estructura del proyecto

```text
Netflix_Catalogo_2021/
├── assets/
│   ├── app_preview.png
│   └── netflix_icon.png
├── data/
│   ├── processed/
│   └── raw/
│       └── netflixData.csv
├── notebooks/
│   └── 01_exploracion_datos.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   └── recommender.py
├── tests/
│   ├── test_data_processing.py
│   └── test_recommender.py
├── app.py
├── environment-conda.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

He separado el proyecto en módulos para evitar concentrar todo el código en el notebook o en la aplicación. El procesamiento de datos, el recomendador, la interfaz y los tests tienen responsabilidades independientes.

## Instalación con Conda

La forma recomendada de crear el entorno es:

```powershell
conda env create -f environment-conda.yml
conda activate Netflix_Cat_2021
```

## Instalación con pip

Para instalar las dependencias necesarias para ejecutar la aplicación:

```powershell
python -m pip install -r requirements.txt
```

Para instalar también las herramientas de desarrollo:

```powershell
python -m pip install -r requirements-dev.txt
```

## Ejecutar la aplicación

Desde la carpeta principal del proyecto:

```powershell
python -m streamlit run app.py
```

La aplicación estará disponible normalmente en:

```text
http://localhost:8501
```

## Tests

He creado tests automatizados para comprobar el procesamiento de datos y el funcionamiento del recomendador.

Para ejecutarlos:

```powershell
python -m pytest -v
```

Actualmente, el proyecto cuenta con siete tests que comprueban:

- La normalización de los textos.
- La carga de archivos CSV.
- La detección de columnas obligatorias ausentes.
- La transformación y preparación de los datos.
- La búsqueda de títulos.
- La generación de recomendaciones.
- La gestión de títulos inexistentes.

## Calidad del código

También utilizo Ruff para detectar posibles problemas de estilo o errores en el código:

```powershell
python -m ruff check app.py src tests
```
## Fuente de los datos

Para desarrollar este proyecto he utilizado el dataset [Netflix Movies and TV Shows 2021](https://www.kaggle.com/datasets/satpreetmakhija/netflix-movies-and-tv-shows-2021), publicado por Satpreet Makhija en Kaggle.

Según la descripción del autor, el catálogo fue recopilado a partir de Flixable, un buscador externo de contenidos de Netflix.

El dataset se distribuye con licencia **CC0: Public Domain**.

## Limitaciones

Durante el desarrollo he identificado algunas limitaciones:

- El catálogo contiene información publicada hasta 2021.
- Algunos títulos tienen datos incompletos.
- El sistema no aprende de las preferencias del usuario.
- Las recomendaciones dependen de los metadatos disponibles.
- La aplicación no incluye carteles, tráilers ni disponibilidad actualizada.

## Posibles mejoras

En futuras versiones me gustaría:

- Añadir filtros por género, año y tipo de contenido.
- Incorporar carteles e información adicional de cada título.
- Comparar diferentes configuraciones de TF-IDF.
- Evaluar otros pesos para las características.
- Incorporar valoraciones o preferencias de usuarios.
- Publicar la aplicación para que pueda utilizarse desde Internet.

## Conclusión

Con este proyecto he puesto en práctica la limpieza y transformación de datos, el procesamiento de lenguaje natural con TF-IDF, la similitud del coseno, la organización modular de código, los tests automatizados y la creación de una interfaz con Streamlit.

Este proyecto forma parte de mi portfolio personal y tiene una finalidad educativa. No está afiliado oficialmente con Netflix.
