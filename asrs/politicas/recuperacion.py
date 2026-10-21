"""Políticas de recuperación: qué ítem del SKU solicitado se retira."""
import numpy as np

from asrs.politicas.restriccion import MatricesRestriccion


class PoliticaRecuperacionFCFS:
    # FCFS estricto: siempre el ítem más antiguo del SKU, aunque haya que reubicar lo que tiene delante
    # Corresponde a un almacén con control de lote o de vencimiento, donde el orden de salida es obligatorio
    nombre = 'FCFS'

    def _masAntiguo(self, instancia_estanteria, escalar_sku, matriz_candidatas):
        # Índice del ítem más antiguo del SKU entre las celdas candidatas, o una tupla vacía si no hay ninguno
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
    # El ítem más antiguo del SKU entre los que no tienen nada delante; solo si ninguno está accesible se retira el más antiguo
    # y se reubica lo que lo bloquea. Corresponde a un almacén sin control de lote, donde todas las unidades de un SKU son intercambiables
    nombre = 'FCFSAccesible'

    def seleccionar(self, instancia_estanteria, escalar_sku):
        matriz_ocupacion = MatricesRestriccion().ocupacion(instancia_estanteria)
        vector_indice = self._masAntiguo(instancia_estanteria, escalar_sku, matriz_ocupacion & MatricesRestriccion().accesibilidad(matriz_ocupacion))
        return vector_indice if vector_indice else self._masAntiguo(instancia_estanteria, escalar_sku, matriz_ocupacion)


# Claves de configuración de las reglas de recuperación
REGLAS_RECUPERACION = {
    'fcfs': PoliticaRecuperacionFCFS,
    'fcfs_accesible': PoliticaRecuperacionFCFSAccesible,
}


def crear_politica_recuperacion(txt_clave):
    if txt_clave not in REGLAS_RECUPERACION:
        raise ValueError(f"Regla de recuperación desconocida: '{txt_clave}'. Disponibles: {', '.join(REGLAS_RECUPERACION)}.")
    return REGLAS_RECUPERACION[txt_clave]()
