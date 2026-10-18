"""Política de recuperación: primero en entrar, primero en salir (FCFS)."""
import numpy as np

from asrs.politicas.restriccion import MatricesRestriccion


class PoliticaRecuperacionFCFS:
    nombre = 'FCFS'

    def seleccionar(self, instancia_estanteria, escalar_etiqueta_clase):
        # Devuelve el índice del ítem más antiguo de la clase solicitada, o una tupla vacía si no hay ninguno
        vector_indice_elegido = ()
        escalar_generacion_minima = None

        for vector_indice in np.argwhere(MatricesRestriccion().ocupacion(instancia_estanteria)):
            vector_indice = tuple(vector_indice)
            instancia_item = instancia_estanteria.matriz_contenedor_items[vector_indice]
            if instancia_item.escalar_etiqueta_clase != escalar_etiqueta_clase:
                continue
            if escalar_generacion_minima is None or instancia_item.escalar_etiqueta_generacion < escalar_generacion_minima:
                escalar_generacion_minima = instancia_item.escalar_etiqueta_generacion
                vector_indice_elegido = vector_indice

        return vector_indice_elegido
