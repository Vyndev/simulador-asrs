"""Política basada en clases: zonas fijas por clase."""
import numpy as np

from asrs.politicas.asignacion import PoliticaAsignacion
from asrs.politicas.zonificacion import construir_zonificacion


class PoliticaBasadaEnClases(PoliticaAsignacion):
    nombre = 'BasadaEnClases'

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, config):
        super().__init__(instancia_estanteria, instancia_grua, instancia_puntoES, config)

        # Las zonas se reparten en el orden de las etiquetas: la clase 1 recibe las celdas más cercanas al punto E/S
        vector_orden_clases = self.ordenClases()
        self.matriz_zonificacion = construir_zonificacion(
            self.tiempoChebyshevXY(instancia_puntoES.vector_posicion),
            config["vector_capacidad_clase"],
            vector_orden_clases)

    def ordenClases(self):
        return list(range(1, len(self.config["vector_capacidad_clase"]) + 1))

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Dentro de la zona de la clase se prefiere la celda más cercana al punto E/S
        return self.tiempoChebyshevXY(self.instancia_puntoES.vector_posicion)
