import pandas as pd
import pytest

from src.data_processing import (
    cargar_datos,
    normalizar_elementos,
    preparar_datos,
)


def crear_dataset_prueba():
    """Crea un catálogo pequeño para realizar los tests."""

    return pd.DataFrame(
        {
            "Show Id": [
                "show-001",
                "show-002",
            ],
            "Title": [
                "Película de prueba",
                "Película de prueba",
            ],
            "Description": [
                "A mysterious adventure in a dangerous city.",
                "A romantic story about friendship.",
            ],
            "Director": [
                None,
                "Jane Doe",
            ],
            "Genres": [
                "Horror Movies, International Movies",
                "Romantic Movies, Dramas",
            ],
            "Cast": [
                None,
                "Actor One, Actor Two",
            ],
            "Production Country": [
                "United States",
                None,
            ],
            "Release Date": [
                2020.0,
                2020.0,
            ],
            "Imdb Score": [
                "7.5/10",
                None,
            ],
            "Content Type": [
                "Movie",
                "Movie",
            ],
        }
    )


def test_normalizar_elementos():
    """Comprueba diferentes casos de normalización."""

    assert normalizar_elementos(
        "Horror Movies, International Movies"
    ) == "horror_movies international_movies"

    assert normalizar_elementos("  Cho Il  ") == "cho_il"
    assert normalizar_elementos("") == ""


def test_cargar_datos(tmp_path):
    """Comprueba que se puede cargar un CSV válido."""

    datos_originales = crear_dataset_prueba()
    ruta_csv = tmp_path / "netflix_prueba.csv"

    datos_originales.to_csv(
        ruta_csv,
        index=False,
    )

    datos_cargados = cargar_datos(ruta_csv)

    assert datos_cargados.shape == datos_originales.shape
    assert "Release Date" in datos_cargados.columns
    assert "Title" in datos_cargados.columns


def test_cargar_datos_con_columna_faltante(tmp_path):
    """Comprueba que se rechaza un CSV incompleto."""

    datos_incompletos = crear_dataset_prueba().drop(
        columns=["Release Date"]
    )

    ruta_csv = tmp_path / "datos_incompletos.csv"

    datos_incompletos.to_csv(
        ruta_csv,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Release Date",
    ):
        cargar_datos(ruta_csv)


def test_preparar_datos():
    """Comprueba las principales transformaciones del catálogo."""

    datos_originales = crear_dataset_prueba()
    datos_limpios = preparar_datos(datos_originales)

    # Se conserva el número de títulos
    assert len(datos_limpios) == len(datos_originales)

    # Release Date se transforma en Release Year
    assert "Release Date" not in datos_limpios.columns
    assert "Release Year" in datos_limpios.columns
    assert str(datos_limpios["Release Year"].dtype) == "Int64"

    # La puntuación IMDb se convierte en un número
    assert datos_limpios.loc[0, "Imdb Score"] == pytest.approx(7.5)

    # Los valores ausentes utilizados por el modelo se convierten en texto vacío
    assert datos_limpios.loc[0, "Director Model"] == ""
    assert datos_limpios.loc[0, "Cast Model"] == ""

    # Se comprueba la normalización
    assert (
        datos_limpios.loc[0, "Genres Model"]
        == "horror_movies international_movies"
    )

    # Se crean las columnas que utilizará el recomendador
    assert "Combined Features" in datos_limpios.columns
    assert "Selection Label" in datos_limpios.columns

    # Las etiquetas deben ser únicas aunque los títulos coincidan
    assert datos_limpios["Selection Label"].is_unique

    # El texto utilizado por TF-IDF no debe contener valores nulos
    assert datos_limpios["Combined Features"].notna().all()