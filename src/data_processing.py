from pathlib import Path

import pandas as pd

# Columnas que necesita la aplicación para funcionar
COLUMNAS_REQUERIDAS = {
    "Show Id",
    "Title",
    "Description",
    "Director",
    "Genres",
    "Cast",
    "Production Country",
    "Release Date",
    "Imdb Score",
    "Content Type",
}


def cargar_datos(ruta_csv: str | Path) -> pd.DataFrame:
    """
    Carga el catálogo de Netflix y comprueba sus columnas.
    """

    ruta_csv = Path(ruta_csv)

    if not ruta_csv.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset: {ruta_csv}"
        )

    datos = pd.read_csv(ruta_csv)

    columnas_faltantes = (
        COLUMNAS_REQUERIDAS - set(datos.columns)
    )

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: "
            f"{sorted(columnas_faltantes)}"
        )

    return datos


def normalizar_elementos(texto: str) -> str:
    """
    Normaliza listas separadas por comas.

    Ejemplo:
    'Horror Movies, International Movies'
    se convierte en:
    'horror_movies international_movies'
    """

    elementos = texto.split(",")

    elementos_normalizados = [
        elemento.strip().lower().replace(" ", "_")
        for elemento in elementos
        if elemento.strip()
    ]

    return " ".join(elementos_normalizados)


def preparar_datos(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y prepara el catálogo para el recomendador.
    """

    datos_limpios = datos.copy()

    # La columna contiene años, no fechas completas
    datos_limpios = datos_limpios.rename(
        columns={"Release Date": "Release Year"}
    )

    datos_limpios["Release Year"] = (
        pd.to_numeric(
            datos_limpios["Release Year"],
            errors="coerce",
        )
        .astype("Int64")
    )

    # Convertimos puntuaciones como "6.6/10" en 6.6
    datos_limpios["Imdb Score"] = pd.to_numeric(
        datos_limpios["Imdb Score"]
        .astype("string")
        .str.replace("/10", "", regex=False),
        errors="coerce",
    )

    # Los campos vacíos no deben introducir términos falsos
    columnas_texto = [
        "Description",
        "Genres",
        "Director",
        "Cast",
        "Production Country",
    ]

    datos_limpios[columnas_texto] = (
        datos_limpios[columnas_texto].fillna("")
    )

    # Versiones normalizadas utilizadas por el modelo
    datos_limpios["Genres Model"] = (
        datos_limpios["Genres"]
        .apply(normalizar_elementos)
    )

    datos_limpios["Director Model"] = (
        datos_limpios["Director"]
        .apply(normalizar_elementos)
    )

    datos_limpios["Cast Model"] = (
        datos_limpios["Cast"]
        .apply(normalizar_elementos)
    )

    datos_limpios["Country Model"] = (
        datos_limpios["Production Country"]
        .apply(normalizar_elementos)
    )

    datos_limpios["Description Model"] = (
        datos_limpios["Description"]
        .str.lower()
        .str.strip()
    )

    # Combinación ponderada de las características
    datos_limpios["Combined Features"] = (
        datos_limpios["Genres Model"] + " "
        + datos_limpios["Genres Model"] + " "
        + datos_limpios["Genres Model"] + " "
        + datos_limpios["Director Model"] + " "
        + datos_limpios["Director Model"] + " "
        + datos_limpios["Cast Model"] + " "
        + datos_limpios["Country Model"] + " "
        + datos_limpios["Description Model"]
    ).str.strip()

    # Etiqueta legible para seleccionar cada título
    anio_texto = (
        datos_limpios["Release Year"]
        .astype("string")
        .fillna("Año desconocido")
    )

    datos_limpios["Selection Label"] = (
        datos_limpios["Title"]
        + " ("
        + anio_texto
        + ", "
        + datos_limpios["Content Type"]
        + ")"
    )

    # Distinguimos las etiquetas que todavía estén repetidas
    etiquetas_duplicadas = (
        datos_limpios["Selection Label"]
        .duplicated(keep=False)
    )

    datos_limpios.loc[
        etiquetas_duplicadas,
        "Selection Label",
    ] = (
        datos_limpios.loc[
            etiquetas_duplicadas,
            "Selection Label",
        ]
        + " ["
        + datos_limpios.loc[
            etiquetas_duplicadas,
            "Show Id",
        ].str[:8]
        + "]"
    )

    return datos_limpios