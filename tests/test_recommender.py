import pandas as pd
import pytest

from src.data_processing import preparar_datos
from src.recommender import RecomendadorNetflix


def crear_catalogo_prueba():
    """Crea un catálogo pequeño para probar el recomendador."""

    return pd.DataFrame(
        {
            "Show Id": [
                "zombie01",
                "zombie02",
                "romance01",
            ],
            "Title": [
                "Zombie City",
                "Zombie Escape",
                "Love in Paris",
            ],
            "Description": [
                "A zombie outbreak in a dangerous city.",
                "Survivors escape from a zombie outbreak.",
                "A romantic love story in Paris.",
            ],
            "Director": [
                "John Horror",
                "John Horror",
                "Jane Romance",
            ],
            "Genres": [
                "Horror Movies, Thrillers",
                "Horror Movies, Thrillers",
                "Romantic Movies",
            ],
            "Cast": [
                "Actor One",
                "Actor One, Actor Two",
                "Actor Three",
            ],
            "Production Country": [
                "United States",
                "United States",
                "France",
            ],
            "Release Date": [
                2020.0,
                2021.0,
                2019.0,
            ],
            "Imdb Score": [
                "7.0/10",
                "7.5/10",
                "8.0/10",
            ],
            "Content Type": [
                "Movie",
                "Movie",
                "Movie",
            ],
        }
    )


def crear_modelo():
    """Prepara los datos y crea el recomendador de prueba."""

    datos = preparar_datos(crear_catalogo_prueba())

    return RecomendadorNetflix(
        datos=datos,
        max_features=100,
    )


def test_buscar_titulos():
    modelo = crear_modelo()

    resultado = modelo.buscar_titulos(
        texto="zombie",
        cantidad=10,
    )

    assert len(resultado) == 2
    assert resultado["Selection Label"].str.contains(
        "Zombie",
        case=False,
    ).all()


def test_recomendar_titulo_similar():
    modelo = crear_modelo()

    etiqueta = modelo.datos.loc[
        modelo.datos["Title"].eq("Zombie City"),
        "Selection Label",
    ].iloc[0]

    resultado = modelo.recomendar(
        etiqueta_titulo=etiqueta,
        cantidad=1,
    )

    assert len(resultado) == 1
    assert resultado.iloc[0]["Title"] == "Zombie Escape"
    assert resultado.iloc[0]["Director"] == "John Horror"
    assert resultado.iloc[0]["Similarity (%)"] > 0


def test_recomendar_titulo_inexistente():
    modelo = crear_modelo()

    with pytest.raises(
        ValueError,
        match="No se encontró",
    ):
        modelo.recomendar(
            etiqueta_titulo="Título inexistente",
            cantidad=5,
        )