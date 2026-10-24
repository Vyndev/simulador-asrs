"""Políticas de recuperación: primero en entrar, primero en salir (FCFS), estricto o entre los ítems accesibles."""
import numpy as np

from asrs.politicas.restriccion import MatricesRestriccion


class PoliticaRecuperacionFCFS:
    # FCFS estricto: siempre sale el ítem más antiguo del SKU, aunque haya que reubicar los que tiene delante
    # Corresponde a los casos en que el orden de salida es obligatorio, por ejemplo perecederos o trazabilidad por lote
    nombre = 'FCFS'

    def _masAntiguo(self, instancia_estanteria, escalar_sku, matriz_candidatas):
        # Devuelve el índice del ítem más antiguo del SKU entre las celdas candidatas, o una tupla vacía si no hay ninguno
        vector_indice_elegido = ()
        escalar_generacion_minima = None

        for vector_indice in np.argwhere(matriz_candidatas):
            vector_indice = tuple(vector_indice)
            instancia_item = instancia_estanteria.matriz_contenedor_items[vector_indice]
            if instancia_item.escalar_sku != escalar_sku:
                continue
            if escalar_generacion_minima is None or instancia_item.escalar_etiqueta_generacion < escalar_generacion_minima:
                escalar_generacion_minima = instancia_item.escalar_etiqueta_generacion
                vector_indice_elegido = vector_indice

        return vector_indice_elegido

    def seleccionar(self, instancia_estanteria, escalar_sku):
        # Devuelve el índice del ítem más antiguo del SKU solicitado, o una tupla vacía si no hay ninguno
        return self._masAntiguo(instancia_estanteria, escalar_sku, MatricesRestriccion().ocupacion(instancia_estanteria))


class PoliticaRecuperacionFCFSAccesible(PoliticaRecuperacionFCFS):
    # FCFS entre accesibles: sale el ítem más antiguo del SKU entre los que no tienen nada delante, así no hay que reubicar
    # Solo si todos los ítems del SKU están bloqueados se recurre al más antiguo de todos, como en FCFS estricto
    nombre = 'FCFS-Accesible'

    def seleccionar(self, instancia_estanteria, escalar_sku):
        matriz_ocupacion = MatricesRestriccion().ocupacion(instancia_estanteria)
        matriz_accesibles = matriz_ocupacion & MatricesRestriccion().accesibilidad(matriz_ocupacion)
        vector_indice_elegido = self._masAntiguo(instancia_estanteria, escalar_sku, matriz_accesibles)
        if len(vector_indice_elegido):
            return vector_indice_elegido
        return self._masAntiguo(instancia_estanteria, escalar_sku, matriz_ocupacion)
