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
