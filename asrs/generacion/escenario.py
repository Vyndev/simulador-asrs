"""Escenario de demanda: la lista de llenado, la lista de operación y la popularidad móvil que genera el modelo de suministro y demanda."""
from dataclasses import dataclass

import numpy as np


@dataclass
class EscenarioDemanda:
    # Todo lo que el generador entrega a la simulación. El SKU de cada tarea es el número del producto, que no se reutiliza
    lista_llenado: list # Tuplas (0, sku): el inventario al final del calentamiento, en orden aleatorio
    lista_operacion: list # Tuplas (actividad, sku), 0 es almacenar y 1 es recuperar, con exactamente escalar_largo_secuencia tareas
    vector_dia_de_tarea: list # Día de cada tarea de la operación, 0 es el 1 de enero del año evaluado
    conjunto_retiros: set # Índices de lista_operacion que son retiros de fin de ciclo
    matriz_popularidad: np.ndarray # (días de operación generados, n): popularidad móvil de cada posición al empezar cada día
    vector_inventario_inicial: np.ndarray # Unidades de cada posición al final del calentamiento, la posición i en el índice i - 1
    resumen: dict # Indicadores del escenario: demanda no atendida, lotes aplazados, retiros, lanzamientos, ocupación, concentración, σ, días y surtido
    vector_dia_lanzamiento: np.ndarray = None # Día de operación en que nace cada producto, negativo si ya existía el día 0
    vector_dia_cierre: np.ndarray = None # Día de operación en que cada producto cierra su ciclo y sale, un valor enorme si no cierra

    def popularidad(self, escalar_dia):
        # Popularidad móvil del día, acotada a la última fila generada. Es lo único que las políticas ven de la demanda
        escalar_ultimo = self.matriz_popularidad.shape[0] - 1
        if escalar_ultimo < 0:
            return np.zeros(self.matriz_popularidad.shape[1], dtype=self.matriz_popularidad.dtype)
        return self.matriz_popularidad[min(max(int(escalar_dia), 0), escalar_ultimo)].copy()

    def activos(self, escalar_dia):
        # Productos que existen el día de operación dado (han nacido y no han cerrado su ciclo), acotado a la última fila generada
        # Es el catálogo del almacén: lo único que las políticas saben de qué SKU existen
        escalar_ultimo = max(self.matriz_popularidad.shape[0] - 1, 0)
        escalar_dia = min(max(int(escalar_dia), 0), escalar_ultimo)
        return (self.vector_dia_lanzamiento <= escalar_dia) & (escalar_dia < self.vector_dia_cierre)
