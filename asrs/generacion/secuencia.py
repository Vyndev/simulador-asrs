"""Generación de secuencias de operaciones (almacenar/recuperar por clase)."""
import random as rnd
import numpy as np
import pandas as pd


class GeneradorSecuencia:
    def __init__(self, escalar_largo_de_secuencia, vector_probabilidad_acumudada_AR,
                 vector_probabilidad_acumulada_A_clasificado, vector_probabilidad_acumulada_R_clasificado):
        # Inicializa el generador de secuencias con una semilla, un largo de secuencia y vectores de probabilidad acumulada
        # Si se proporciona una semilla, se utiliza para inicializar el generador de números aleatorios

        # Establece los atributos de la instancia con los valores proporcionados
        self.escalar_largo_de_secuencia = escalar_largo_de_secuencia
        self.vector_probabilidad_acumudada_AR = vector_probabilidad_acumudada_AR
        self.vector_probabilidad_acumulada_A_clasificado = vector_probabilidad_acumulada_A_clasificado
        self.vector_probabilidad_acumulada_R_clasificado = vector_probabilidad_acumulada_R_clasificado

    def probabilidadesSencillas(self, vector_probabilidades_acumuladas):

        # Este método comobierte las probabilidades acumumladas en probabilidades sencillas
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
        vector_proporciones = np.asarray(vector_proporciones, dtype=float)
        vector_cuotas = vector_proporciones * escalar_total
        vector_base = np.floor(vector_cuotas + 1e-9).astype(int)
        escalar_faltantes = int(escalar_total - vector_base.sum())
        if escalar_faltantes > 0:
            vector_orden = np.argsort(-(vector_cuotas - vector_base), kind='stable')
            for escalar_indice in vector_orden[:escalar_faltantes]:
                vector_base[escalar_indice] += 1
        return vector_base

    def generarSecuencia(self):
        # Genera una secuencia de operaciones
        vector_prop_sencillas_AR = self.probabilidadesSencillas(
            self.vector_probabilidad_acumudada_AR)

        escalar_cantidad_almacenamientos, escalar_cantidad_recuperaciones = self.repartir(
            self.escalar_largo_de_secuencia, vector_prop_sencillas_AR)

        vector_etiquetas_clase_almacenamiento = np.arange(
            1, len(self.vector_probabilidad_acumulada_A_clasificado) + 1)
        vector_etiquetas_clase_recuperacion = np.arange(
            1, len(self.vector_probabilidad_acumulada_R_clasificado) + 1)

        vector_cantidades_A = self.repartir(
            escalar_cantidad_almacenamientos,
            self.probabilidadesSencillas(self.vector_probabilidad_acumulada_A_clasificado))
        vector_almacenamientos_clasificados = []
        for escalar_etiqueta, escalar_repeticiones in zip(vector_etiquetas_clase_almacenamiento, vector_cantidades_A):
            vector_almacenamientos_clasificados.extend(
                [int(escalar_etiqueta)] * int(escalar_repeticiones))

        vector_cantidades_R = self.repartir(
            escalar_cantidad_recuperaciones,
            self.probabilidadesSencillas(self.vector_probabilidad_acumulada_R_clasificado))
        vector_recuperaciones_clasificadas = []
        for escalar_etiqueta, escalar_repeticiones in zip(vector_etiquetas_clase_recuperacion, vector_cantidades_R):
            vector_recuperaciones_clasificadas.extend(
                [int(escalar_etiqueta)] * int(escalar_repeticiones))

        vector_ceros = [0] * len(vector_almacenamientos_clasificados)
        vector_unos = [1] * len(vector_recuperaciones_clasificadas)

        matriz_lista_almacenamientos = list(
            zip(vector_ceros, vector_almacenamientos_clasificados))
        matriz_lista_recuperaciones = list(
            zip(vector_unos, vector_recuperaciones_clasificadas))
        matriz_lista_operaciones = matriz_lista_almacenamientos
        matriz_lista_operaciones.extend(matriz_lista_recuperaciones)
        rnd.shuffle(matriz_lista_operaciones)
        self.matriz_lista_operaciones_sorteada = matriz_lista_operaciones
        # Devuelve la matriz de operaciones
        return self.matriz_lista_operaciones_sorteada

    def __repr__(self):
        # Convierte la matriz de operaciones a un DataFrame de pandas y la devuelve como una cadena
        # Esto proporciona una representación legible de la matriz de operaciones
        return pd.DataFrame(self.matriz_lista_operaciones_sorteada, columns=['Actividad', 'Clase']).to_string(index=False)
