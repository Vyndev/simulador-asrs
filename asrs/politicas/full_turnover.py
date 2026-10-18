"""Política full-turnover (Hausman, Graves y Schwarz, 1976)."""
import numpy as np

from asrs.politicas.basada_en_clases import PoliticaBasadaEnClases


class PoliticaFullTurnover(PoliticaBasadaEnClases):
    nombre = 'FullTurnover'

    def ordenClases(self):
        # Las clases se ordenan de mayor a menor demanda y la de mayor demanda recibe las ubicaciones fijas más cercanas
        # A igual demanda se conserva el orden de las etiquetas
        vector_demanda = np.asarray(self.config["vector_demanda_clase"], dtype=float)
        return [int(escalar_indice) + 1 for escalar_indice in np.argsort(-vector_demanda, kind='stable')]
