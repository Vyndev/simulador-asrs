"""Política aleatoria: el canal se elige al azar."""
import random as rnd

import numpy as np

from asrs.politicas.asignacion import PoliticaAsignacion


class PoliticaAleatoria(PoliticaAsignacion):
    nombre = 'Aleatoria'

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Con minimización de varianza cada canal recibe un valor aleatorio y las capas de un mismo canal comparten el valor:
        # se sortea el canal y la regla elige luego la celda libre más profunda de él
        # Sin ella cada celda recibe su propio valor aleatorio: se elige al azar cualquier celda libre accesible
        escalar_filas, escalar_columnas, escalar_capas = self.instancia_estanteria.matriz_contenedor_items.shape
        generador = np.random.default_rng(rnd.getrandbits(64))
        if not self.config.get("booleano_minimizacion_varianza", True):
            return generador.random((escalar_filas, escalar_columnas, escalar_capas))
        matriz_canales = generador.random((escalar_filas, escalar_columnas))

        return np.repeat(matriz_canales[:, :, np.newaxis], escalar_capas, axis=2)
