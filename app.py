from pathlib import Path

import streamlit as st

from src.data_processing import cargar_datos, preparar_datos
from src.recommender import RecomendadorNetflix

# Rutas del proyecto
RAIZ_PROYECTO = Path(__file__).resolve().parent

RUTA_DATOS = (
    RAIZ_PROYECTO
    / "data"
    / "raw"
    / "netflixData.csv"
)

RUTA_ICONO = (
    RAIZ_PROYECTO
    / "assets"
    / "netflix_icon.png"
)


# Configuración general de la página
st.set_page_config(
    page_title="Recomendador de Netflix",
    page_icon=str(RUTA_ICONO),
    layout="wide",
)

# Estilos visuales de la aplicación
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(255, 75, 80, 0.13),
                transparent 32%
            ),
            radial-gradient(
                circle at 85% 85%,
                rgba(120, 0, 20, 0.10),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #0e1117 0%,
                #0b0d12 50%,
                #111318 100%
            );
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(
    show_spinner="Preparando el recomendador..."
)
def cargar_recomendador():
    """
    Carga los datos, los prepara y construye el recomendador.

    El resultado queda guardado en memoria para no reconstruir
    el modelo cada vez que el usuario interactúa con la aplicación.
    """

    datos_originales = cargar_datos(RUTA_DATOS)
    datos_preparados = preparar_datos(datos_originales)

    return RecomendadorNetflix(
        datos=datos_preparados,
        max_features=20_000,
    )


# Encabezado de la aplicación
columna_icono, columna_titulo = st.columns(
    [1, 11],
    vertical_alignment="center",
)

with columna_icono:
    st.image(
        str(RUTA_ICONO),
        width=80,
    )

with columna_titulo:
    st.title("Recomendador de Netflix")

st.write(
    "Selecciona una película o serie y descubre títulos similares "
    "según su género, reparto, director, país y descripción."
)


# Cargamos el recomendador y controlamos posibles errores
try:
    recomendador = cargar_recomendador()

except (FileNotFoundError, ValueError) as error:
    st.error(
        f"No se pudo cargar el recomendador: {error}"
    )
    st.stop()


# Información del catálogo
cantidad_titulos = f"{len(recomendador.datos):,}".replace(",", ".")

st.caption(
    f"Catálogo disponible: {cantidad_titulos} títulos."
)

st.divider()
st.subheader("Encuentra títulos similares")

# Opciones disponibles en el selector
etiquetas_titulos = (
    recomendador.datos["Selection Label"]
    .sort_values()
    .tolist()
)


# Controles de búsqueda
columna_titulo, columna_cantidad = st.columns(
    [4, 1]
)

with columna_titulo:
    titulo_seleccionado = st.selectbox(
        label="Película o serie",
        options=etiquetas_titulos,
        index=None,
        placeholder="Escribe o selecciona un título...",
    )

with columna_cantidad:
    cantidad_recomendaciones = st.slider(
        label="Recomendaciones",
        min_value=3,
        max_value=10,
        value=5,
    )


boton_recomendar = st.button(
    label="Obtener recomendaciones",
    type="primary",
    width="stretch",
)

if boton_recomendar:
    if titulo_seleccionado is None:
        st.warning(
            "Selecciona una película o serie antes de continuar."
        )

    else:
        try:
            recomendaciones = recomendador.recomendar(
                etiqueta_titulo=titulo_seleccionado,
                cantidad=cantidad_recomendaciones,
            )

        except ValueError as error:
            st.error(str(error))

        else:
            st.divider()
            st.subheader("Recomendaciones")

            # Traducimos los nombres solo para mostrarlos
            recomendaciones_mostradas = recomendaciones.rename(
                columns={
                    "Title": "Título",
                    "Content Type": "Tipo",
                    "Release Year": "Año",
                    "Genres": "Géneros",
                    "Imdb Score": "IMDb",
                    "Similarity (%)": "Similitud (%)",
                }
            )

            st.dataframe(
                recomendaciones_mostradas,
                hide_index=True,
                width="stretch",
                column_config={
                    "Año": st.column_config.NumberColumn(
                        format="%d"
                    ),
                    "IMDb": st.column_config.NumberColumn(
                        format="%.1f"
                    ),
                    "Similitud (%)": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    ),
                },
            )

            st.caption(
                "La similitud representa la cercanía entre las características "
                "de los títulos. No es una probabilidad de que el contenido "
                "vaya a gustarle al usuario."
            )

st.divider()

with st.expander("¿Cómo funciona el recomendador?"):
    st.markdown(
        """
        El sistema compara los títulos mediante sus características:

        - Géneros.
        - Director.
        - Reparto.
        - País de producción.
        - Descripción.

        Los textos se convierten en vectores numéricos mediante **TF-IDF**.
        Después se utiliza la **similitud del coseno** para localizar los
        títulos con características más parecidas.

        Este es un recomendador basado en contenido, por lo que no utiliza
        historiales de visualización ni información personal de los usuarios.
        """
    )