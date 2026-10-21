"""Política basada en clases: los SKUs se agrupan en clases y cada clase tiene una zona fija."""
import numpy as np

from asrs.generacion.probabilidades import perfil_inventario
from asrs.politicas.asignacion import PoliticaAsignacion
from asrs.politicas.zonificacion import clasificar_skus, construir_zonificacion


class PoliticaBasadaEnClases(PoliticaAsignacion):
    nombre = 'BasadaEnClases'

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, config):
        super().__init__(instancia_estanteria, instancia_grua, instancia_puntoES, config)

        # La agrupación es parte de la política: los SKUs más demandados forman la clase 1 y reciben las celdas más cercanas al punto E/S
        vector_demanda = np.asarray(config["vector_demanda_sku"], dtype=float)
        self.vector_clase_de_sku = clasificar_skus(vector_demanda, config)

        # El espacio de cada clase es proporcional al inventario de sus SKUs (según el perfil de inventario), no a su demanda
        escalar_clases = int(self.vector_clase_de_sku.max())
        vector_inventario = perfil_inventario(vector_demanda, config["texto_perfil_inventario"])
        vector_espacio = np.array([vector_inventario[self.vector_clase_de_sku == clase].sum() for clase in range(1, escalar_clases + 1)])
        self.matriz_zonificacion = construir_zonificacion(
            self.tiempoChebyshevXY(instancia_puntoES.vector_posicion),
            vector_espacio,
            list(range(1, escalar_clases + 1)))

    def etiquetaZona(self, escalar_sku):
        return int(self.vector_clase_de_sku[escalar_sku - 1])

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Dentro de la zona de la clase se prefiere la celda más cercana al punto E/S
        return self.tiempoChebyshevXY(self.instancia_puntoES.vector_posicion)
