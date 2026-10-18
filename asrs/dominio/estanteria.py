"""Estantería con celdas organizadas en filas, columnas y capas."""
import numpy as np

from asrs.dominio.item import Item


class Estanteria:
    def __init__(self, vector_posicion, vector_configuracion, vector_cotas, vector_centro_celda) -> None:
        # 'vector_posicion' especifica las coordenadas de la esquina inferior izquierda más profunda de la estantería.
        self.vector_posicion = vector_posicion

        # 'vector_configuracion' es un arreglo que contiene la configuración de la estantería en términos de número de filas, columnas y capas.
        self.vector_configuracion = vector_configuracion

        # 'vector_cotas' contiene las dimensiones de cada celda (alto, ancho, profundidad).
        self.vector_cotas = vector_cotas

        # 'vector_centro_celda' indica la posición relativa del ítem dentro de la celda, descrita en proporciones de las dimensiones de la celda.
        self.vector_centro_celda = vector_centro_celda

        # Calcula la capacidad total de la estantería multiplicando las dimensiones de 'vector_configuracion'.
        self.escalar_capacidad_total = np.prod(vector_configuracion)

        # Inicializa una matriz tridimensional para almacenar instancias de 'Item' según la configuración dada.
        self.matriz_contenedor_items = np.zeros(
            vector_configuracion, dtype=Item)

        # Caché de las posiciones de las celdas, se calcula la primera vez que se consulta
        self._matriz_posiciones_celdas = None

    def posicionesCeldas(self):
        # Devuelve una matriz (filas, columnas, capas, 3) con la posición global del punto de referencia de cada celda
        if self._matriz_posiciones_celdas is None:
            matriz_indices = np.indices(tuple(self.vector_configuracion)).transpose(1, 2, 3, 0)
            vector_cotas = np.asarray(self.vector_cotas, dtype=float)
            vector_desplazamiento = vector_cotas * np.asarray(self.vector_centro_celda, dtype=float)
            self._matriz_posiciones_celdas = (matriz_indices * vector_cotas
                                              + np.asarray(self.vector_posicion, dtype=float)
                                              + vector_desplazamiento)
        return self._matriz_posiciones_celdas

    def __repr__(self) -> str:
        # Método especial que define cómo se representa una instancia de 'Estanteria' como una cadena de texto.
        # Se muestra la posición de la estantería, la capacidad ocupada y la capacidad total.
        self.capacidad_ocupada = np.count_nonzero(self.matriz_contenedor_items)
        return f'Posición: {self.vector_posicion} | Capacidad ocupada: {self.capacidad_ocupada}/{self.escalar_capacidad_total}'
