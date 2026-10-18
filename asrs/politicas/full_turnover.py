"""Política full-turnover (Hausman, Graves y Schwarz, 1976): cada SKU tiene su propia zona fija."""
import numpy as np

from asrs.politicas.asignacion import PoliticaAsignacion
from asrs.politicas.zonificacion import construir_zonificacion


class PoliticaFullTurnover(PoliticaAsignacion):
    nombre = 'FullTurnover'

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, config):
        super().__init__(instancia_estanteria, instancia_grua, instancia_puntoES, config)

        # Los SKUs se ordenan de mayor a menor demanda y el más demandado recibe las ubicaciones fijas más cercanas
        # A igual demanda se conserva el orden por número de SKU. El espacio de cada SKU es proporcional a su demanda
        vector_demanda = np.asarray(config["vector_demanda_sku"], dtype=float)
        vector_orden_skus = [int(escalar_indice) + 1 for escalar_indice in np.argsort(-vector_demanda, kind='stable')]
        self.matriz_zonificacion = construir_zonificacion(
            self.tiempoChebyshevXY(instancia_puntoES.vector_posicion),
            vector_demanda,
            vector_orden_skus)

    def etiquetaZona(self, escalar_sku):
        # Cada SKU es su propia zona
        return int(escalar_sku)

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Dentro de la zona del SKU se prefiere la celda más cercana al punto E/S
        return self.tiempoChebyshevXY(self.instancia_puntoES.vector_posicion)
