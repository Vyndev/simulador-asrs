"""Política basada en clases: los SKUs se agrupan en clases y cada clase tiene una zona fija."""
import numpy as np

from asrs.politicas.asignacion import PoliticaAsignacion
from asrs.politicas.zonificacion import agrupar_skus_en_clases, construir_zonificacion


class PoliticaBasadaEnClases(PoliticaAsignacion):
    nombre = 'BasadaEnClases'

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, config):
        super().__init__(instancia_estanteria, instancia_grua, instancia_puntoES, config)

        # La agrupación es parte de la política: los SKUs más demandados forman la clase 1 y reciben las celdas más cercanas al punto E/S
        vector_demanda = np.asarray(config["vector_demanda_sku"], dtype=float)
        self.vector_clase_de_sku = agrupar_skus_en_clases(vector_demanda, config["vector_proporcion_skus_por_clase"])

        # El espacio de cada clase es proporcional a la demanda de sus SKUs, porque el inventario sigue a la demanda
        escalar_clases = int(self.vector_clase_de_sku.max())
        vector_espacio = np.array([vector_demanda[self.vector_clase_de_sku == clase].sum() for clase in range(1, escalar_clases + 1)])
        self.matriz_zonificacion = construir_zonificacion(
            self.tiempoChebyshevXY(instancia_puntoES.vector_posicion),
            vector_espacio,
            list(range(1, escalar_clases + 1)))

    def etiquetaZona(self, escalar_sku):
        return int(self.vector_clase_de_sku[escalar_sku - 1])

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Dentro de la zona de la clase se prefiere la celda más cercana al punto E/S
        return self.tiempoChebyshevXY(self.instancia_puntoES.vector_posicion)
