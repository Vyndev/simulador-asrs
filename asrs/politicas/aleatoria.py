"""Política aleatoria: el canal se elige al azar."""
import random as rnd

import numpy as np

from asrs.politicas.asignacion import PoliticaAsignacion


class PoliticaAleatoria(PoliticaAsignacion):
    nombre = 'Aleatoria'

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Cada canal recibe un valor aleatorio y las capas de un mismo canal comparten el valor:
        # se sortea el canal, pero la regla de minimización de varianza actúa antes: el sorteo es entre los canales con la capa libre más profunda
        escalar_filas, escalar_columnas, escalar_capas = self.instancia_estanteria.matriz_contenedor_items.shape
        generador = np.random.default_rng(rnd.getrandbits(64))
        matriz_canales = generador.random((escalar_filas, escalar_columnas))

        return np.repeat(matriz_canales[:, :, np.newaxis], escalar_capas, axis=2)
