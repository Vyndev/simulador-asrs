"""Generación de la lista de tareas: almacenar o recuperar un SKU."""
import random as rnd
import numpy as np
import pandas as pd

from asrs.generacion.probabilidades import demanda_de_etapa
from asrs.utilidades import repartir_entero


class GeneradorSecuencia:
    # La lista de tareas solo conoce SKUs. Agrupar los SKUs en clases, si hace falta, es decisión de cada política
    def __init__(self, escalar_largo_de_secuencia, vector_probabilidad_acumulada_AR, vector_probabilidad_acumulada_sku):
        # vector_probabilidad_acumulada_AR: probabilidad acumulada de [almacenar, recuperar]
        # vector_probabilidad_acumulada_sku: probabilidad acumulada de la demanda de cada SKU, el SKU 1 es el primero
        self.escalar_largo_de_secuencia = escalar_largo_de_secuencia
        self.vector_probabilidad_acumulada_AR = vector_probabilidad_acumulada_AR
        self.vector_probabilidad_acumulada_sku = vector_probabilidad_acumulada_sku

    def probabilidadesSencillas(self, vector_probabilidades_acumuladas):
        # Este método convierte las probabilidades acumuladas en probabilidades sencillas
        escalar_cantidad_elementos = len(vector_probabilidades_acumuladas)
        vector_probabilidades_sencillas = np.zeros(escalar_cantidad_elementos)
        vector_probabilidades_sencillas[0] = vector_probabilidades_acumuladas[0]

        for escalar_indice in range(1, escalar_cantidad_elementos):
            vector_probabilidades_sencillas[escalar_indice] = vector_probabilidades_acumuladas[escalar_indice] - \
                vector_probabilidades_acumuladas[escalar_indice - 1]

        return vector_probabilidades_sencillas

    @staticmethod
    def repartir(escalar_total, vector_proporciones):
        # Reparte un total entero según proporciones con el método del mayor resto, la suma siempre es igual al total
        return repartir_entero(escalar_total, vector_proporciones)

    def generarSecuencia(self):
        # Genera la lista de tareas como pares (actividad, SKU): actividad 0 es almacenar y 1 es recuperar
        # Las cantidades son exactas (mayor resto) y el azar solo decide el orden
        escalar_cantidad_almacenamientos, escalar_cantidad_recuperaciones = self.repartir(
            self.escalar_largo_de_secuencia, self.probabilidadesSencillas(self.vector_probabilidad_acumulada_AR))

        vector_proporciones_sku = self.probabilidadesSencillas(self.vector_probabilidad_acumulada_sku)

        # Cada tarea de cada tipo se reparte entre los SKUs según su demanda
        matriz_lista_operaciones = []
        for escalar_actividad, escalar_cantidad in ((0, escalar_cantidad_almacenamientos), (1, escalar_cantidad_recuperaciones)):
            vector_cantidades = self.repartir(escalar_cantidad, vector_proporciones_sku) if escalar_cantidad else np.zeros(len(vector_proporciones_sku), dtype=int)
            for escalar_sku, escalar_repeticiones in enumerate(vector_cantidades, start=1):
                matriz_lista_operaciones.extend([(escalar_actividad, escalar_sku)] * int(escalar_repeticiones))

        rnd.shuffle(matriz_lista_operaciones)
        self.matriz_lista_operaciones_sorteada = matriz_lista_operaciones
        return self.matriz_lista_operaciones_sorteada

    def __repr__(self):
        # Convierte la matriz de operaciones a un DataFrame de pandas y la devuelve como una cadena
        return pd.DataFrame(self.matriz_lista_operaciones_sorteada, columns=['Actividad', 'SKU']).to_string(index=False)


def largos_de_etapas(escalar_largo_total, lista_etapas):
    # Cantidad de tareas de cada etapa: reparte el total según el peso de cada una (mayor resto)
    return [int(escalar) for escalar in repartir_entero(escalar_largo_total, [dic_etapa["peso"] for dic_etapa in lista_etapas])]


def generar_secuencia_por_etapas(escalar_largo_total, lista_etapas, vector_demanda_base):
    # Demanda variable: la lista se arma por etapas consecutivas y cada una tiene su propia mezcla de almacenar y recuperar
    # y su propia demanda por SKU. El orden se sortea dentro de cada etapa, las etapas se recorren en orden
    matriz_lista_operaciones = []
    for dic_etapa, escalar_largo in zip(lista_etapas, largos_de_etapas(escalar_largo_total, lista_etapas)):
        if escalar_largo == 0:
            continue
        vector_demanda = demanda_de_etapa(vector_demanda_base, dic_etapa["sigma"], dic_etapa["rotacion"])
        vector_prob_acum_sku = np.cumsum(vector_demanda)
        vector_prob_acum_sku[-1] = 1.0
        vector_prob_acum_AR = np.array([dic_etapa["prob_almacenar"], 1.0])
        matriz_lista_operaciones.extend(GeneradorSecuencia(escalar_largo, vector_prob_acum_AR, vector_prob_acum_sku).generarSecuencia())
    return matriz_lista_operaciones
