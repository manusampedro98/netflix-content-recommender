import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecomendadorNetflix:
    """
    Recomendador de contenido basado en TF-IDF
    y similitud del coseno.
    """

    def __init__(
        self,
        datos: pd.DataFrame,
        max_features: int = 20_000,
    ):
        """
        Entrena el recomendador con el catálogo preparado.
        """

        columnas_necesarias = {
            "Title",
            "Selection Label",
            "Combined Features",
            "Content Type",
            "Release Year",
            "Genres",
            "Imdb Score",
        }

        columnas_faltantes = (
            columnas_necesarias - set(datos.columns)
        )

        if columnas_faltantes:
            raise ValueError(
                "Faltan columnas preparadas: "
                f"{sorted(columnas_faltantes)}"
            )

        # Reiniciamos el índice para mantenerlo alineado
        # con las filas de la matriz TF-IDF
        self.datos = (
            datos
            .reset_index(drop=True)
            .copy()
        )

        # Las etiquetas deben identificar un único contenido
        if self.datos["Selection Label"].duplicated().any():
            raise ValueError(
                "Selection Label contiene valores duplicados."
            )

        # Configuramos y entrenamos el vectorizador
        self.vectorizador = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
        )

        self.matriz_tfidf = (
            self.vectorizador.fit_transform(
                self.datos["Combined Features"]
            )
        )

        # Relacionamos cada etiqueta con su posición
        self.indice_por_etiqueta = pd.Series(
            self.datos.index,
            index=self.datos["Selection Label"],
        )

    def buscar_titulos(
        self,
        texto: str,
        cantidad: int = 10,
    ) -> pd.DataFrame:
        """
        Busca títulos que contengan el texto indicado.
        """

        coincidencias = self.datos["Title"].str.contains(
            texto,
            case=False,
            na=False,
            regex=False,
        )

        columnas_resultado = [
            "Selection Label",
            "Genres",
            "Imdb Score",
        ]

        return (
            self.datos
            .loc[coincidencias, columnas_resultado]
            .head(cantidad)
            .copy()
        )

    def recomendar(
        self,
        etiqueta_titulo: str,
        cantidad: int = 5,
    ) -> pd.DataFrame:
        """
        Devuelve los contenidos más similares al seleccionado.
        """

        if cantidad < 1:
            raise ValueError(
                "La cantidad debe ser mayor que cero."
            )

        if etiqueta_titulo not in self.indice_por_etiqueta:
            raise ValueError(
                f"No se encontró el título: {etiqueta_titulo}"
            )

        indice_titulo = int(
            self.indice_por_etiqueta.loc[etiqueta_titulo]
        )

        # Comparamos el título con todo el catálogo
        similitudes = cosine_similarity(
            self.matriz_tfidf[indice_titulo],
            self.matriz_tfidf,
        ).flatten()

        # Ordenamos desde la mayor similitud
        indices_ordenados = similitudes.argsort()[::-1]

        # Excluimos el contenido utilizado como referencia
        indices_recomendados = indices_ordenados[
            indices_ordenados != indice_titulo
        ][:cantidad]

        columnas_resultado = [
            "Title",
            "Content Type",
            "Release Year",
            "Genres",
            "Imdb Score",
        ]

        recomendaciones = (
            self.datos
            .iloc[indices_recomendados][columnas_resultado]
            .copy()
        )

        recomendaciones["Similarity (%)"] = (
            similitudes[indices_recomendados] * 100
        ).round(2)

        return recomendaciones