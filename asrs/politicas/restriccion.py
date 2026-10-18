"""Matrices de restricción dinámicas: ocupación y accesibilidad de las celdas."""
import numpy as np


class MatricesRestriccion:

    def ocupacion(self, instancia_estanteria):  # Matriz dinámica
        # Evalúa el estado de ocupacion de la instancia de estantería, se asigna el valor True al estado ocupado y False al estado libre
        # Las celdas libres contienen el entero 0, las ocupadas una instancia de 'Item'
        return np.not_equal(instancia_estanteria.matriz_contenedor_items, 0).astype(bool)

    def accesibilidad(self, matriz_ocupacion):  # Matriz dinámica
        # Una celda es accesible si ninguna de las celdas delante de ella (capas con índice mayor) está ocupada
        # Se usa para que las asignaciones no atraviesen ítems que obstruyen
        matriz_ocupacion = matriz_ocupacion.astype(int)
        matriz_acumulado_frente = np.flip(np.cumsum(np.flip(matriz_ocupacion, axis=2), axis=2), axis=2) - matriz_ocupacion

        return matriz_acumulado_frente == 0
