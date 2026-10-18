"""Distribuciones de demanda por SKU."""
import numpy as np


class _GeneradorProbAcumuladas:
    # Esta clase se utiliza para generar probabilidades acumuladas.

    def hausman(self, escalar_cantidad_opciones, escalar_sigma):
        # Este método implementa la distribución acumulada de Hausman.

        # Inicializa un arreglo vacío de numpy para almacenar las probabilidades acumuladas.
        vector_probabilidad_acumulada = np.array((), dtype=float)

        # Itera sobre el rango de 1 hasta la cantidad de opciones (inclusivo).
        for escalar_indice_prob_acum in range(1, escalar_cantidad_opciones + 1):
            # Calcula la probabilidad acumulada para cada opción 'i'.
            # La probabilidad se calcula como (i / total_de_opciones) ^ sigma.
            vector_probabilidad_acumulada = np.append(vector_probabilidad_acumulada,
                                                      pow(escalar_indice_prob_acum / escalar_cantidad_opciones,
                                                          escalar_sigma))

        # Devuelve el vector de probabilidades acumuladas.
        return vector_probabilidad_acumulada


def demanda_hausman(escalar_cantidad_skus, escalar_sigma):
    # Demanda de cada SKU según la curva de Hausman: F(i) = (i / n) ** sigma
    # El SKU 1 es el más demandado. Con sigma = 1 la demanda es uniforme y cuanto menor es sigma, más se concentra en pocos SKUs
    vector_acumulada = _GeneradorProbAcumuladas().hausman(escalar_cantidad_skus, escalar_sigma)
    return np.diff(np.concatenate(([0.0], vector_acumulada)))


def demanda_de_etapa(vector_demanda_base, escalar_sigma=None, escalar_rotacion=0):
    # Demanda de cada SKU durante una etapa de demanda variable
    # Sin sigma se parte de la demanda base y con sigma se usa la curva de Hausman de la misma cantidad de SKUs
    # La rotación desplaza el ranking: con rotación k el SKU i recibe la demanda que tenía el SKU i - k, así cambia cuáles SKUs son los más demandados
    vector_base = np.asarray(vector_demanda_base, dtype=float)
    vector_demanda = vector_base if escalar_sigma is None else demanda_hausman(len(vector_base), escalar_sigma)
    return np.roll(vector_demanda, int(escalar_rotacion))
