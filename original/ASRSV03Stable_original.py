import random as rnd
import numpy as np
import pandas as pd
from statistics import stdev
from itertools import product

import copy
import csv
import multiprocessing as mp
import time


import numpy as np

class OperacionesMatrices:
    def enumerar(self, matriz_priorizacion, booleno_invertido=False):
        # Se consulta la forma de la matriz de priorización
        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = matriz_priorizacion.shape
        
        # Se construye una lista en base a la matriz de priorización en donde se consignan los valores de cada elemento y su índice de celda en la estantería
        matriz_lista_valores_x_coordenadas = []
        for escalar_indice_fila in range(escalar_cantidad_filas):
            for escalar_indice_columna in range(escalar_cantidad_columnas):
                for escalar_indice_capa in range(escalar_cantidad_capas):
                    vector_indices = (escalar_indice_fila, escalar_indice_columna, escalar_indice_capa)
                    if matriz_priorizacion[vector_indices] != 0:  # Las celdas con valor de cero se excluyen
                        matriz_lista_valores_x_coordenadas.append((matriz_priorizacion[vector_indices], vector_indices))
                        
        # Se ordena la lista de acuerdo a los valores de los elementos
        matriz_lista_valores_x_coordenadas = sorted(matriz_lista_valores_x_coordenadas, key=lambda x: x[0], reverse=booleno_invertido)
            
        # Se inicializa una matriz vacía con la misma forma que la matriz de priorización, esta servirá para consignar los valores enumerados      
        matriz_priorizacion_enumerada = np.zeros_like(matriz_priorizacion, dtype=int)
        
        # Se recorre la lista ordenada enumerada, se asigna la numeración a la coordenada correspondiente
        # Se inicializa una variable para almacenar en memoria el valor actual del grado en el recorrido
        escalar_memoria_grado = None
        
        # Se inicializa el contador de la enumeración
        escalar_contador = 0
        for escalar_grado, vector_valor_x_coordenada in matriz_lista_valores_x_coordenadas:
            # Si el grado es igual al anterior entonces se asigna el mismo contador
            if escalar_grado == escalar_memoria_grado:
                matriz_priorizacion_enumerada[vector_valor_x_coordenada] = escalar_contador
            # En cambio si es diferente, entonces se incrementa el contador en 1 y se asigna el valor actualizado
            else:
                escalar_contador += 1
                escalar_memoria_grado = escalar_grado
                matriz_priorizacion_enumerada[vector_valor_x_coordenada] = escalar_contador
            
        return matriz_priorizacion_enumerada

    
    def desempate(self, matriz_priorizacion):
        
        # Se consulta la forma de la matriz de priorización
        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = matriz_priorizacion.shape

        # Se recupera el valor correspondiente a la cantidad total de elemetos dentro de la matriz
        escalar_cantidad_elementos = matriz_priorizacion.size

        # Se obtiene una vector aplanado en donde cada elemento está numerado
        vector_matriz_aleatoria_aplanada = np.arange(start = 1, stop = escalar_cantidad_elementos + 1)

        # Se aletoriza el orden de los elementos
        rnd.shuffle(vector_matriz_aleatoria_aplanada)

        # Se escala el vector por un factor de 1 / (escalar_cantidad elementos + 1), así se genera una matriz de elementos aletorios menores que uno sin posibilidad de repetición
        vector_matriz_aleatoria_aplanada = vector_matriz_aleatoria_aplanada / (escalar_cantidad_elementos + 1)

        # Se reconstruye el vector, en un matriz de dimensiones iguales a la matriz de entrada del método
        matriz_aletoria = np.reshape(vector_matriz_aleatoria_aplanada, matriz_priorizacion.shape)

        # Devuelve una nueva matriz de priorización con ruido aplicado, con la observacion de que los elementos con valor 0 deberan ser restaurados
        matriz_priorizacion_preliminar = matriz_priorizacion + matriz_aletoria
        
        # Para restaurar los valores 0s se toma la matriz de priorización original de la cual se deriva una matriz booleana con los elementos activos
        matriz_booleana = (matriz_priorizacion != 0).astype(int)

        # Se multiplica la matriz de priorización y la matriz booleana, esto con el fin de obtener una forma sin elementos repetidos y con unicamente los elementos activos
        matriz_priorizacion = matriz_booleana * matriz_priorizacion_preliminar
        
        return self.enumerar(matriz_priorizacion)
    
    def superponer(self, *matrices_priorizacion):
        matriz_superpuesta = 0
        for matriz_priorizacion in matrices_priorizacion:
            matriz_superpuesta += matriz_priorizacion
        
        return self.enumerar(matriz_superpuesta)
    
    def multiplicar(self, *matrices_priorizacion):
        matriz_superpuesta = 0
        for matriz_priorizacion in matrices_priorizacion:
            matriz_superpuesta *= matriz_priorizacion
        
        return self.enumerar(matriz_superpuesta)
        

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


class Metricas:

    def __init__(self):
        # Contadores generales para las operaciones de almacenamiento, recuperación y reubicacion
        self.cantidad_almacenamientos = 0
        self.cantidad_recuperaciones = 0
        self.cantidad_reubicaciones = 0
        self.cantidad_operaciones = 0

        # Contadores para las reubicaciones durante el almacenamiento y la recuperación
        self.cantidad_reubicaciones_almacenamiento = 0
        self.cantidad_reubicaciones_recuperacion = 0

        # Contadores para las operaciones de reubicación desagregadas
        self.cantidad_reubicaciones_regulares_almacenamiento = 0
        self.cantidad_reubicaciones_predeterminadas_almacenamiento = 0
        self.cantidad_reubicaciones_regulares_recuperacion = 0
        self.cantidad_reubicaciones_predeterminadas_recuperacion = 0

        # Contadores para el tiempo total de almacenamiento, recuperación y reubicacion
        self.tiempo_total_almacenamiento = 0
        self.tiempo_total_recuperacion = 0
        self.tiempo_total_reubicacion = 0

        # Contadores para el tiempo de operaciones de reubicacion durante el almacenamiento y la recuperación
        self.tiempo_total_reubicaciones_almacenamiento = 0
        self.tiempo_total_reubicaciones_recuperacion = 0
        self.tiempo_total_reubicaciones_regulares_almacenamiento = 0
        self.tiempo_total_reubicaciones_predeterminadas_almacenamiento = 0
        self.tiempo_total_reubicaciones_regulares_recuperacion = 0
        self.tiempo_total_reubicaciones_predeterminadas_recuperacion = 0
        self.tiempo_total_operacion = 0

        # Vectores de datos
        self.vector_tiempo_almacenamientos = []
        self.vector_tiempo_recuperaciones = []
        self.vector_tiempo_reubicaciones = []
        self.vector_tiempo_reubicaciones_almacenamiento = []
        self.vector_tiempo_reubicaciones_recuperacion = []
        self.vector_tiempo_reubicaciones_regulares_almacenamiento = []
        self.vector_tiempo_reubicaciones_predeterminadas_almacenamiento = []
        self.vector_tiempo_reubicaciones_regulares_recuperacion = []
        self.vector_tiempo_reubicaciones_predeterminadas_recuperacion = []

    def recuperarInformcion(self):
        self.cantidad_operaciones = self.cantidad_almacenamientos + \
            self.cantidad_recuperaciones
        self.tiempo_total_operacion = self.tiempo_total_almacenamiento + \
            self.tiempo_total_recuperacion + self.tiempo_total_reubicacion

        O0, O1 = self.cantidad_operaciones, self.tiempo_total_operacion
        A0, A1 = self.cantidad_almacenamientos, self.tiempo_total_almacenamiento
        B0, B1 = self.cantidad_recuperaciones, self.tiempo_total_recuperacion
        C0, C1 = self.cantidad_reubicaciones, self.tiempo_total_reubicacion
        D0, D1 = self.cantidad_reubicaciones_almacenamiento, self.tiempo_total_reubicaciones_almacenamiento
        E0, E1 = self.cantidad_reubicaciones_recuperacion, self.tiempo_total_reubicaciones_recuperacion
        F0, F1 = self.cantidad_reubicaciones_regulares_almacenamiento, self.tiempo_total_reubicaciones_regulares_almacenamiento
        G0, G1 = self.cantidad_reubicaciones_predeterminadas_almacenamiento, self.tiempo_total_reubicaciones_predeterminadas_almacenamiento
        H0, H1 = self.cantidad_reubicaciones_regulares_recuperacion, self.tiempo_total_reubicaciones_regulares_recuperacion
        I0, I1 = self.cantidad_reubicaciones_predeterminadas_recuperacion, self.tiempo_total_reubicaciones_predeterminadas_recuperacion

        A2 = stdev(self.vector_tiempo_almacenamientos) if self.vector_tiempo_almacenamientos and len(self.vector_tiempo_almacenamientos) > 1 else 0
        B2 = stdev(self.vector_tiempo_recuperaciones) if self.vector_tiempo_recuperaciones and len(self.vector_tiempo_recuperaciones) > 1 else 0
        C2 = stdev(self.vector_tiempo_reubicaciones) if self.vector_tiempo_reubicaciones and len(self.vector_tiempo_reubicaciones) > 1 else 0
        D2 = stdev(self.vector_tiempo_reubicaciones_almacenamiento) if self.vector_tiempo_reubicaciones_almacenamiento and len(self.vector_tiempo_reubicaciones_almacenamiento) > 1 else 0
        E2 = stdev(self.vector_tiempo_reubicaciones_recuperacion) if self.vector_tiempo_reubicaciones_recuperacion and len(self.vector_tiempo_reubicaciones_recuperacion) > 1 else 0
        F2 = stdev(self.vector_tiempo_reubicaciones_regulares_almacenamiento) if self.vector_tiempo_reubicaciones_regulares_almacenamiento and len(
self.vector_tiempo_reubicaciones_regulares_almacenamiento) > 1 else 0
        G2 = stdev(self.vector_tiempo_reubicaciones_predeterminadas_almacenamiento) if self.vector_tiempo_reubicaciones_predeterminadas_almacenamiento and len(self.vector_tiempo_reubicaciones_predeterminadas_almacenamiento) > 1 else 0
        H2 = stdev(
            self.vector_tiempo_reubicaciones_regulares_recuperacion) if self.vector_tiempo_reubicaciones_regulares_recuperacion and len(
            self.vector_tiempo_reubicaciones_regulares_recuperacion) > 1 else 0
        I2 = stdev(
            self.vector_tiempo_reubicaciones_predeterminadas_recuperacion) if self.vector_tiempo_reubicaciones_predeterminadas_recuperacion and len(
            self.vector_tiempo_reubicaciones_predeterminadas_recuperacion) > 1 else 0

        return [A0, A1, A2, B0, B1, B2, C0, C1, C2, D0, D1, D2, E0, E1, E2, F0, F1, F2, G0, G1, G2, H0, H1, H2, I0, I1, I2, O0, O1]

    def tiempoDezplamiento(self, vector_posicion_inicial, vector_posicion_final, vector_velocidad):
        vector_distancia = np.abs(
            vector_posicion_final - vector_posicion_inicial)

        # Determinar la distancia en el plano xy
        vector_distancia_xy = vector_distancia[:2]

        # Determina la distancia en el tramo z
        escalar_distancia_z = vector_distancia[2]

        # Calcula tiempos de Chebyshev el tramo xz
        vector_tiempo_xy = vector_distancia_xy / vector_velocidad[:2]
        esclar_tiempo_chebyshev_xy = np.max(vector_tiempo_xy)

        # Calcula tiempo de desplazamiento en z
        escalar_tiempo_z = escalar_distancia_z / vector_velocidad[2]

        # Calcula el tiempo total de ejecución
        tiempo_total_ejecucion = esclar_tiempo_chebyshev_xy + escalar_tiempo_z

        # Retorna el tiempo total de ejecución
        return tiempo_total_ejecucion

    def registrarAlmacenamiento(self, tiempo):
        # Agregar el tiempo de almacenamiento al DataFrame
        self.cantidad_almacenamientos += 1
        self.tiempo_total_almacenamiento += tiempo
        self.vector_tiempo_almacenamientos.append(tiempo)

    def registrarRecuperacion(self, tiempo):
        # Agregar el tiempo de recuperación al DataFrame
        self.cantidad_recuperaciones += 1
        self.tiempo_total_recuperacion += tiempo
        self.vector_tiempo_recuperaciones.append(tiempo)

    def registrarReubicacion(self, tiempo):
        # Agregar el tiempo de reubicación al DataFrame
        self.cantidad_reubicaciones += 1
        self.tiempo_total_reubicacion += tiempo
        self.vector_tiempo_reubicaciones.append(tiempo)

    def registrarReubicacionAlmacenamiento(self, tiempo):
        # Agregar el tiempo de reubicación durante el almacenamiento al DataFrame
        self.cantidad_reubicaciones_almacenamiento += 1
        self.tiempo_total_reubicaciones_almacenamiento += tiempo
        self.vector_tiempo_reubicaciones_almacenamiento.append(tiempo)

    def registrarReubicacionRecuperacion(self, tiempo):
        # Agregar el tiempo de reubicación durante la recuperación al DataFrame
        self.cantidad_reubicaciones_recuperacion += 1
        self.tiempo_total_reubicaciones_recuperacion += tiempo
        self.vector_tiempo_reubicaciones_recuperacion.append(tiempo)

    def registrarReubicacionRegularAlmacenamiento(self, tiempo):
        # Agregar el tiempo de reubicación regular durante el almacenamiento al DataFrame
        self.cantidad_reubicaciones_regulares_almacenamiento += 1
        self.tiempo_total_reubicaciones_regulares_almacenamiento += tiempo
        self.vector_tiempo_reubicaciones_regulares_almacenamiento.append(
            tiempo)

    def registrarReubicacionPredeterminadaAlmacenamiento(self, tiempo):
        # Agregar el tiempo de reubicación predeterminada durante el almacenamiento al DataFrame
        self.cantidad_reubicaciones_predeterminadas_almacenamiento += 1
        self.tiempo_total_reubicaciones_predeterminadas_almacenamiento += tiempo
        self.vector_tiempo_reubicaciones_predeterminadas_almacenamiento.append(
            tiempo)

    def registrarReubicacionRegularRecuperacion(self, tiempo):
        # Agregar el tiempo de reubicación regular durante la recuperación al DataFrame
        self.cantidad_reubicaciones_regulares_recuperacion += 1
        self.tiempo_total_reubicaciones_regulares_recuperacion += tiempo
        self.vector_tiempo_reubicaciones_regulares_recuperacion.append(tiempo)

    def registrarReubicacionPredeterminadaRecuperacion(self, tiempo):
        # Agregar el tiempo de reubicación predeterminada durante la recuperación al DataFrame
        self.cantidad_reubicaciones_predeterminadas_recuperacion += 1
        self.tiempo_total_reubicaciones_predeterminadas_recuperacion += tiempo
        self.vector_tiempo_reubicaciones_predeterminadas_recuperacion.append(
            tiempo)

class Item:
    escalar_contador_global_generacion = 0

    # Se definen los atributos constructores
    def __init__(self, escalar_etiqueta_clase) -> None:
        # Se aumenta en 1 el contador de ítems generados
        Item.escalar_contador_global_generacion += 1

        # Se guarda la etiqueta de generación
        self.escalar_etiqueta_generacion = Item.escalar_contador_global_generacion

        # Se guarda la etiqueta de clase
        self.escalar_etiqueta_clase = escalar_etiqueta_clase

    # Se configura la forma en que se representa la instancia en el momento que se llama
    def __repr__(self) -> str:
        return f'{self.escalar_etiqueta_clase}|{self.escalar_etiqueta_generacion}'


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

    def __repr__(self) -> str:
        # Método especial que define cómo se representa una instancia de 'Estanteria' como una cadena de texto.
        # Se muestra la posición de la estantería, la capacidad ocupada y la capacidad total.
        self.capacidad_ocupada = np.count_nonzero(self.matriz_contenedor_items)
        return f'Posición: {self.vector_posicion} | Capacidad ocupada: {self.capacidad_ocupada}/{self.escalar_capacidad_total}'


class Grua:
    def __init__(self, vector_posicion_reposo, vector_velocidad, tiempo_cargadescarga):
        # Inicializa la grúa con una posición de reposo y una velocidad
        # La posición actual de la grúa se establece en la posición de reposo
        # La grúa inicialmente no tiene ninguna carga ni está en ninguna estantería o punto E/S
        self.vector_posicion_resposo = vector_posicion_reposo
        self.vector_posicion_actual = vector_posicion_reposo
        self.vector_velocidad = vector_velocidad
        self.vector_carga = []
        self.instancia_estanteria_actual = 0
        self.instancia_puntoES_actual = 0
        self.tiempo_cargadescarga = tiempo_cargadescarga
        self.intancia_metricas = Metricas()

    # Métodos para interactuar con la estantería

    def irAEsquinaEstanteria(self, instancia_estanteria):
        # La grúa se mueve a una estantería dada
        # La posición actual de la grúa se actualiza al primer
        vector_posicion_inicial = self.vector_posicion_actual

        self.instancia_puntoES_actual = 0
        self.instancia_estanteria_actual = instancia_estanteria
        self.vector_posicion_actual = np.copy(
            self.instancia_estanteria_actual.vector_posicion)
        self.volverAplano()
        vector_posicion_final = self.vector_posicion_actual

        return self.intancia_metricas.tiempoDezplamiento(vector_posicion_inicial, vector_posicion_final, self.vector_velocidad)

    def moverEnEstanteria(self, vector_indice_destino):
        # La grúa se mueve dentro de la estantería a una posición dada
        # La posición actual de la grúa se actualiza a la nueva posición
        vector_posicion_inicial = self.vector_posicion_actual
        vector_posicion_final = (np.array(vector_indice_destino) * np.copy(self.instancia_estanteria_actual.vector_cotas)) + (
            np.copy(self.instancia_estanteria_actual.vector_cotas) * np.copy(
                self.instancia_estanteria_actual.vector_centro_celda))

        self.vector_posicion_actual = vector_posicion_final

        # Retorna el tiempo total de ejecución
        return self.intancia_metricas.tiempoDezplamiento(vector_posicion_inicial, vector_posicion_final, self.vector_velocidad)

    def cargarItemDeEstanteria(self, vector_coordenada_recuperacion):
        # La grúa recoge un ítem de una posición dada en la estantería
        # El ítem se añade a la carga de la grúa y se elimina de la estantería
        instancia_item = self.instancia_estanteria_actual.matriz_contenedor_items[
            vector_coordenada_recuperacion]
        self.vector_carga.append(instancia_item)
        self.instancia_estanteria_actual.matriz_contenedor_items[vector_coordenada_recuperacion] = 0

        return self.tiempo_cargadescarga

    def soltarItemEnEstanteria(self, vector_coordenada_almacenamiento):
        # La grúa suelta un ítem en una posición dada en la estantería
        # El ítem se elimina de la carga de la grúa y se añade a la estantería
        instancia_item = self.vector_carga.pop()
        self.instancia_estanteria_actual.matriz_contenedor_items[
            vector_coordenada_almacenamiento] = instancia_item

        return self.tiempo_cargadescarga

    # Métodos para interactuar con el punto E/S

    def irAPuntoES(self, instancia_PuntoES):
        # La grúa se mueve a un punto E/S dado
        # La posición actual de la grúa se actualiza a la posición del punto E/S

        vector_posicion_inicial = self.vector_posicion_actual
        self.instancia_estanteria_actual = 0
        self.instancia_puntoES_actual = instancia_PuntoES
        vector_posicion_final = self.instancia_puntoES_actual.vector_posicion
        self.vector_posicion_actual = vector_posicion_final

        # Retorna el tiempo total de ejecución
        return self.intancia_metricas.tiempoDezplamiento(vector_posicion_inicial, vector_posicion_final, self.vector_velocidad)

    def cargarItemPuntoES(self, indice=-1):
        # La grúa recoge el ítem del punto E/S en la posición dada por 'indice'
        # Si 'indice' no se proporciona, se recoge el último ítem (comportamiento de pop())
        # El ítem se elimina del punto E/S y se carga en la grúa
        instancia_item = self.instancia_puntoES_actual.matriz_contenedor_temporal_items.pop(
            indice)
        self.vector_carga.append(instancia_item)

        return self.tiempo_cargadescarga

    def soltarItemEnPuntoES(self):
        # La grúa suelta un ítem en el punto E/S
        # El ítem se añade al final del contenedor temporal del punto E/S
        instancia_item = self.vector_carga.pop()
        self.instancia_puntoES_actual.matriz_contenedor_temporal_items.append(
            instancia_item)

        return self.tiempo_cargadescarga

    def soltarItemEnPuntoESSalida(self):
        # La grúa suelta un ítem en el punto E/S
        # El ítem se añade al final del contenedor de salida del punto E/S
        instancia_item = self.vector_carga.pop()
        self.instancia_puntoES_actual.matriz_contenedor_salida_items.append(
            instancia_item)

        return self.tiempo_cargadescarga

    # Métodos para reposicionar la grúa

    def volverAplano(self):
        # La grúa se mueve a la altura del riel (posición z de reposo)
        vector_posicion_inicial = self.vector_posicion_actual
        self.vector_posicion_actual[2] = self.vector_posicion_resposo[2]
        vector_posicion_final = self.vector_posicion_actual

        # Retorna el tiempo total de ejecución
        return self.intancia_metricas.tiempoDezplamiento(vector_posicion_inicial, vector_posicion_final, self.vector_velocidad)

    def volverAReposo(self):
        # La grúa vuelve a su posición de reposo
        # La grúa ya no está en ninguna estantería o punto E/S
        vector_posicion_inicial = self.vector_posicion_actual
        vector_posicion_final = self.vector_posicion_resposo

        self.instancia_estanteria_actual = 0
        self.instancia_puntoES_actual = 0
        self.vector_posicion_actual = vector_posicion_final

        # Retorna el tiempo total de ejecución
        return self.intancia_metricas.tiempoDezplamiento(vector_posicion_inicial, vector_posicion_final, self.vector_velocidad)

    def __repr__(self) -> str:
        # Este método devuelve una representación en cadena de caracteres del objeto
        # Incluye la posición actual de la grúa y su velocidad
        return f'{self.vector_posicion_actual}|{self.vector_velocidad}'


class PuntoES:
    def __init__(self, vector_posicion) -> None:
        # Inicializa el punto E/S con una posición dada
        # La posición se guarda en 'vector_posicion'
        self.vector_posicion = vector_posicion

        # Inicializa un contenedor temporal y de salida para los ítems
        # Inicialmente los contenedores son listas vacías
        self.matriz_contenedor_temporal_items = []
        self.matriz_contenedor_salida_items = []

    def __repr__(self) -> str:
        # Este método devuelve una representación en cadena de caracteres del objeto
        # La cadena incluye la posición del punto E/S y los ítems en el contenedor temporal
        return f'{self.vector_posicion}|{self.matriz_contenedor_temporal_items}'


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

    def generarSecuencia(self):
        # Genera una secuencia de operaciones
        vector_prop_sencillas_AR = self.probabilidadesSencillas(
            self.vector_probabilidad_acumudada_AR)

        escalar_cantidad_almacenamientos = int(
            vector_prop_sencillas_AR[0] * self.escalar_largo_de_secuencia)
        escalar_cantidad_recuperaciones = int(
            vector_prop_sencillas_AR[1] * self.escalar_largo_de_secuencia)

        vector_etiquetas_clase_almacenamiento = np.arange(
            1, len(self.vector_probabilidad_acumulada_A_clasificado) + 1)
        vector_etiquetas_clase_recuperacion = np.arange(
            1, len(self.vector_probabilidad_acumulada_R_clasificado) + 1)

        vector_prop_sencillas_A_clasificado = self.probabilidadesSencillas(
            self.vector_probabilidad_acumulada_A_clasificado)
        vector_almacenamientos_clasificados = []
        for escalar_etiqueta, escalar_proporcion in zip(vector_etiquetas_clase_almacenamiento, vector_prop_sencillas_A_clasificado):
            escalar_repeticiones = int(
                escalar_cantidad_almacenamientos * escalar_proporcion)
            vector_almacenamientos_clasificados.extend(
                [escalar_etiqueta] * escalar_repeticiones)

        vector_prop_sencillas_A_clasificado = self.probabilidadesSencillas(
            self.vector_probabilidad_acumulada_R_clasificado)
        vector_recuperaciones_clasificadas = []
        for escalar_etiqueta, escalar_proporcion in zip(vector_etiquetas_clase_recuperacion, vector_prop_sencillas_A_clasificado):
            escalar_repeticiones = int(
                escalar_cantidad_recuperaciones * escalar_proporcion)
            vector_recuperaciones_clasificadas.extend(
                [escalar_etiqueta] * escalar_repeticiones)

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


class MatricesPriorizacion:

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES) -> None:

        self.instancia_estanteria = instancia_estanteria
        self.instancia_grua = instancia_grua
        self.instancia_puntoES = instancia_puntoES
        
    def A(self):
        # Priorización no obstructiva
        # Esta matriz de prirización tiene sentido en recuperación
        # Calcula la matriz de priorización basado en el atributo "escalar_etiqueta_clase",
        # se asigna menor prioridad a las celdas vacías,
        # los ítems con menor de valor atributo obtienen la mayor prioridad, en este caso se obedece a la política FCFS

        # Obtiene la matriz de ítems y sus dimensiones.
        matriz_contenedor_items = self.instancia_estanteria.matriz_contenedor_items
        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = matriz_contenedor_items.shape

        # Inicializa la matriz de valores de atributos y la señal para celdas vacías.
        matriz_valor_atributo = np.zeros((escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas))

        # Itera sobre cada celda en la matriz de items.
        for escalar_indice_filas in range(escalar_cantidad_filas):
            for escalar_indice_columnas in range(escalar_cantidad_columnas):
                for escalar_indice_capas in range(escalar_cantidad_capas):
                    vector_indice_celda = escalar_indice_filas, escalar_indice_columnas, escalar_indice_capas
                    # Si la celda contiene un ítem, obtiene el valor del atributo.
                    if isinstance(matriz_contenedor_items[vector_indice_celda], Item):
                        instancia_item = matriz_contenedor_items[vector_indice_celda]
                        matriz_valor_atributo[vector_indice_celda] = getattr(instancia_item, "escalar_etiqueta_clase")

        # Enumera los elementos de la matriz de forma ascendente, con el fin de guiar la recuperación empezando por los ítems de ingreso temprano.
        matriz_priorizacion_A = OperacionesMatrices().enumerar(matriz_valor_atributo)

        return matriz_priorizacion_A

    def O(self):
        # Priorización canal cercano al origen
        # Calcula la matriz de priorización para el tiempo mínimo de Chebyshev desde el punto E/S a cualquier canal,
        # Entendiendose que los canales más cercanos al punto E/S tienen mayor prioridad.

        # Obtiene la posicion de referencia(PuntoES) y de la estantería.
        vector_posicion_referencia = self.instancia_puntoES.vector_posicion
        vector_posicion_estanteria = self.instancia_estanteria.vector_posicion

        # Calcula el vector de posición desde la grúa a la estantería.
        vector_referencia_estanteria = vector_posicion_estanteria - vector_posicion_referencia

        # Obtiene los datos de la configuración y las cotas de la estantería.
        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = self.instancia_estanteria.vector_configuracion
        vector_cotas = self.instancia_estanteria.vector_cotas
        vector_centro_celda = self.instancia_estanteria.vector_centro_celda

        # Calcula punto de referencia dentro de la celda en términos relativos a las cotas.
        vector_desplazamiento_celda = vector_centro_celda * vector_cotas

        # Llena 'matriz_posiciones_celdas' con las coordenadas de cada celda.
        matriz_posiciones_celdas = np.zeros((escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas, 3), dtype=float)
        for escalar_indice_filas in range(escalar_cantidad_filas):
            for escalar_indice_columnas in range(escalar_cantidad_columnas):
                for escalar_indice_capas in range(escalar_cantidad_capas):
                    # Calcula los vectores de posición de cada celda.
                    vector_indice_celda = escalar_indice_filas, escalar_indice_columnas, escalar_indice_capas
                    matriz_posiciones_celdas[vector_indice_celda] = (vector_indice_celda * vector_cotas) + vector_referencia_estanteria + vector_desplazamiento_celda

        # Calcula el tiempo Chebyshev para el tramo xy.
        matriz_distancias_xy = np.abs(matriz_posiciones_celdas[:, :, :, :2])

        # Calcula los tiempos de viaje para cada componente xy.
        vector_velocidad_xy = self.instancia_grua.vector_velocidad[:2]
        matriz_tiempos_xy = matriz_distancias_xy / vector_velocidad_xy

        # Calcula el tiempo de viaje Chebyshev desde la grúa a todos los canales.
        matriz_tiempos_chebyshev_xy = np.max(matriz_tiempos_xy, axis=3)

        # Construye la matriz de priorización enumerando los elementos de forma ascendente, siendo los elementos más cercanos los de mayor prioridad.
        matriz_priorizacion_N = OperacionesMatrices().enumerar(matriz_tiempos_chebyshev_xy)

        return matriz_priorizacion_N

    def P(self):
        # Prioización de profundidad(Minimización de varianza)
        # Este bloque calcula la matriz de priorización, apuntando a las capas mas profundas,
        # es decir que las capas más profundas tienen mayor prioridad

        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = self.instancia_estanteria.vector_configuracion
        matriz_capas_celdas = np.zeros((escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas), dtype=float)

        for escalar_indice_filas in range(escalar_cantidad_filas):
            for escalar_indice_columnas in range(escalar_cantidad_columnas):
                for escalar_indice_capas in range(escalar_cantidad_capas):
                    vector_indice_celda = escalar_indice_filas, escalar_indice_columnas, escalar_indice_capas
                    matriz_capas_celdas[vector_indice_celda] = escalar_indice_capas + 1

        matriz_priorizacion_P = OperacionesMatrices().enumerar(matriz_capas_celdas)

        return matriz_priorizacion_P
    
    def S(self):
        # Prioización de profundidad(Minimización de varianza)
        # Este bloque calcula la matriz de priorización, apuntando a las capas mas profundas,
        # es decir que las capas más profundas tienen mayor prioridad

        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = self.instancia_estanteria.vector_configuracion
        matriz_capas_celdas = np.zeros((escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas), dtype=float)

        for escalar_indice_filas in range(escalar_cantidad_filas):
            for escalar_indice_columnas in range(escalar_cantidad_columnas):
                for escalar_indice_capas in reversed(range(escalar_cantidad_capas)):
                    vector_indice_celda = escalar_indice_filas, escalar_indice_columnas, escalar_indice_capas
                    matriz_capas_celdas[vector_indice_celda] = escalar_indice_capas + 1

        matriz_priorizacion_P = OperacionesMatrices().enumerar(matriz_capas_celdas)
        
        return matriz_priorizacion_P
        

    def T(self):
        # Priorización canal cercano a la grúa
        # Calcula la matriz de priorización para el tiempo mínimo de Chebyshev desde la posición actual de la grúa a los canales de la estantería

        # Obtiene la posiciones de referencia y de la estantería.
        vector_posicion_referencia = self.instancia_grua.vector_posicion_actual
        vector_posicion_estanteria = self.instancia_estanteria.vector_posicion

        # Calcula el vector de posición desde la grúa a la estantería.
        vector_referencia_estanteria = vector_posicion_estanteria - vector_posicion_referencia

        # Obtiene los datos de la configuración y las cotas de la estantería.
        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = self.instancia_estanteria.vector_configuracion
        vector_cotas = self.instancia_estanteria.vector_cotas
        vector_centro_celda = self.instancia_estanteria.vector_centro_celda

        # Calcula la posición en términos de medidas.
        vector_desplazamiento_celda = vector_centro_celda * vector_cotas

        # Llena 'matriz_posiciones_celdas' con las coordenadas de cada celda.
        matriz_posiciones_celdas = np.zeros((escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas, 3), dtype=float)
        for escalar_indice_filas in range(escalar_cantidad_filas):
            for escalar_indice_columnas in range(escalar_cantidad_columnas):
                for escalar_indice_capas in range(escalar_cantidad_capas):
                    # Calcula los vectores de posición de cada celda.
                    vector_indice_celda = escalar_indice_filas, escalar_indice_columnas, escalar_indice_capas
                    matriz_posiciones_celdas[vector_indice_celda] = (vector_indice_celda * vector_cotas) + vector_referencia_estanteria + vector_desplazamiento_celda

        # Calcula los tiempos por cada tramo de forma separada.

        # Calcula el tiempo Chebyshev para el tramo xy.
        matriz_distancias_xy = np.abs(matriz_posiciones_celdas[:, :, :, :2])

        # Calcula los tiempos de viaje para cada componente xy.
        vector_velocidad_xy = self.instancia_grua.vector_velocidad[:2]
        matriz_tiempos_xy = matriz_distancias_xy / vector_velocidad_xy

        # Calcula el tiempo de viaje Chebyshev desde la grúa a todas las celdas.
        matriz_tiempos_chebyshev_xy = np.max(matriz_tiempos_xy, axis=3)

        # Construye la matriz de priorización numerando los elementos de forma ascendente.
        matriz_priorizacion_T = OperacionesMatrices().enumerar(matriz_tiempos_chebyshev_xy)
        
        return matriz_priorizacion_T

    def U(self):
        # Esta matriz de priorización tiene sentido en almacenamiento y reubicación
        # Calcula la matriz de priorización basado en el atributo "escalar_etiqueta_generacion",
        # se asigna mayor prioridad a las celdas vacías,
        # los items con mayor valor de atributo obtienen la mayor prioridad.
        # Esto implica que se prioriza realizar el almacenamiento o recuperación en primer lugar sobre celdas vacías y luego evitando obstruir ítems próximos a salir

        # Obtiene la matriz de items y sus dimensiones.
        matriz_contenedor_items = self.instancia_estanteria.matriz_contenedor_items
        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = matriz_contenedor_items.shape

        # Inicializa la matriz de valores de atributos con -1 (señal para celdas vacías).
        matriz_valor_atributo = np.full((escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas), -1)

        # Itera sobre cada celda en la matriz de items.
        for escalar_indice_filas in range(escalar_cantidad_filas):
            for escalar_indice_columnas in range(escalar_cantidad_columnas):
                for escalar_indice_capas in range(escalar_cantidad_capas - 1):  # -1 para evitar el índice fuera de rango
                    vector_indice_celda = escalar_indice_filas, escalar_indice_columnas, escalar_indice_capas
                    # Si la celda contiene un ítem y la celda de arriba está vacía, asigna el valor del atributo a la celda de arriba.
                    if isinstance(matriz_contenedor_items[vector_indice_celda], Item) and not isinstance(matriz_contenedor_items[vector_indice_celda[0], vector_indice_celda[1], vector_indice_celda[2] + 1], Item):
                        instancia_item = matriz_contenedor_items[vector_indice_celda]
                        matriz_valor_atributo[vector_indice_celda] = 0 # Se excluye el almacenamiento/reubicación a las celdas ocupuadas
                        # Se asigna el valor de la etiqueta de entrada a la posición de acceso en el canal
                        matriz_valor_atributo[vector_indice_celda[0], vector_indice_celda[1], vector_indice_celda[2] + 1] = getattr(instancia_item, "escalar_etiqueta_generacion")
        
        # Se obtiene el valor máximo de etiqueta de entrada y se le suma uno, esto con el fin de garantizar que sea el nuevo máximo     
        escalar_entrada_maxima = np.max(matriz_valor_atributo) + 1
        
        # A las celdas vacías se le asgina el valor del nuevo máximo recien calculado
        matriz_valor_atributo[matriz_valor_atributo == -1] = escalar_entrada_maxima
        
        # Enumera los elementos de la matriz de la matriz en forma descendente, guiando la reubiación sobre celdas vacías y procurando no obstruir ítems próximos a salir
        matriz_priorizacion_U = OperacionesMatrices().enumerar(matriz_valor_atributo, booleno_invertido = True)

        return matriz_priorizacion_U

    def X(self):
        # Se obtiene una matriz de priorizacion aleatoria

        # Obtiene la matriz de contenedor de items y su capacidad total
        matriz_contenedor_items = self.instancia_estanteria.matriz_contenedor_items
        escalar_capacidad_total = self.instancia_estanteria.escalar_capacidad_total

        # Obtiene las dimensiones de la matriz de contenedor de items
        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = matriz_contenedor_items.shape

        # Crea un vector de secuencia aleatoria de 1 a capacidad total
        vector_secuencia_aleatoria = [escalar_aleatorio for escalar_aleatorio in range(1, escalar_capacidad_total + 1)]

        # Desordena el vector de secuencia aleatoria
        rnd.shuffle(vector_secuencia_aleatoria)

        # Reorganiza el vector desordenado en una matriz con las mismas dimensiones que la matriz de contenedor de items
        matriz_priorizacion_X = np.reshape(vector_secuencia_aleatoria, (escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas))

        # Retorna la matriz de priorización
        return matriz_priorizacion_X
    
    def K(self):
        
        # Priorización de profundidad(minización de varianza)
        matriz_priorizacion_P = self.P()
        
        # Priorización no obstructiva
        matriz_priorizacion_U = self.U()
        
        # Prioización no obstructiva con minimización de varianza
        matriz_priorizacion_K = OperacionesMatrices().superponer(matriz_priorizacion_P, matriz_priorizacion_U)
        
        return matriz_priorizacion_K

    def Q(self):
        # Politica de minimización de varianza priorizando cercanía al origen
        # Los canales más cercanos al punto E/S y las posiciones mas profundas tienen mayor prioriridad
        
        # Priorización de profundidad(minización de varianza)
        matriz_priorizacion_P = self.P()
        
        
        # Priorización canal cercano al origen
        matriz_priorizacion_O = self.O()
        
        # Priorización minimización de varianza y cercanía al origen
        matriz_priorizacion_Q = OperacionesMatrices().superponer(matriz_priorizacion_P, matriz_priorizacion_O)
        
        return matriz_priorizacion_Q

    def G(self):
        # Politica de minimización de varianza priorizando cercanía a la posición actual de la grúa
        # Los canales más cercanos a la posición de la grúa y las posiciones mas profundas tienen mayor prioridad
        
        # Priorización de profundidad(minización de varianza)
        matriz_priorizacion_P = self.P()
        
        # Priorización canal cercano a la grúa
        matriz_priorizacion_T = self.T()
        
        matriz_priorizacion_G = OperacionesMatrices().superponer(matriz_priorizacion_P, matriz_priorizacion_T)
        
        return matriz_priorizacion_G
    
    def Z(self):
        # Se define política unicamente para construir la zonificación
        #
        return OperacionesMatrices().superponer(self.O(), self.S())
        
    
    def devolverMatriz(self, txt_alias_matriz):
        # Método 'getter' de convenienza, retorna la matriz de priorización dado su alias
        matriz_priorizacion = getattr(self, txt_alias_matriz)
        
        return matriz_priorizacion()


class MatricesRestriccion:

    def zonificacion(self, matriz_priorizacion, vector_proporcion_clases):  # Es una matriz de restricción fija
        # Se contruye en función de una matriz de priorización dada
        # Recibe un vector que indica en proporiciones la cantidad de ubicaciones a asignar a cada clase
        # Se considera una matriz de restricción fija ya que una vez se construye esta no cambia

        # Se inicializa una matriz de ceros con el mismo tamaño de la matriz de priorización
        matriz_zonificacion = np.zeros_like(matriz_priorizacion, dtype=int)
        
        # Se secuperan las dimesiones de la matriz de priorización
        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = matriz_priorizacion.shape

        
        # Se crea una matriz que contendra el grado extraido de la matriz de priorización y el respectivo índice de celda
        matriz_lista_grado_x_indices = []
        for escalar_indice_filas in range(escalar_cantidad_filas):
            for escalar_indice_columnas in range(escalar_cantidad_columnas):
                for escalar_indice_capas in range(escalar_cantidad_capas):
                    vector_indice_celda = escalar_indice_filas, escalar_indice_columnas, escalar_indice_capas
                    matriz_lista_grado_x_indices.append([matriz_priorizacion[vector_indice_celda], vector_indice_celda])

        # Se organiza el listado de menor a mayor de acuerdo al grado
        matriz_lista_ordenada_grado_x_coordendas = sorted(matriz_lista_grado_x_indices, key=lambda x: x[0])
        
        # Se construye una nueva lista que unicamente contiene los índices de celda, pero ya organizados
        matriz_lista_ordenada_indices = [vector_indice_celda for escalar_priorizacion, vector_indice_celda in matriz_lista_ordenada_grado_x_coordendas]

        # Se calcula la cantidad de celdas a asignar por cada clase, para ellos se opera el vector de proporciones con la cantidad total ubicaciones en la estantería
        vector_celdas_por_clase = vector_proporcion_clases * matriz_priorizacion.size
        # Se redondean valores decimales
        vector_celdas_por_clase = [int(numero) for numero in vector_celdas_por_clase]
        
        # Si la cantidad de celdas asignadas en menor a la capacidad total, de forma predeterminada se asignan las celdas sobrantes a la primera clase
        # Dicha decisión tiene un menor impacto en cuanto la cantidad de celdas disponibles en estanterias sea superior
        if sum(vector_celdas_por_clase) < matriz_priorizacion.size:
            vector_celdas_por_clase[0] = vector_celdas_por_clase[0] + matriz_priorizacion.size - sum(vector_celdas_por_clase)

        # Se construye la matriz de zonificación
        # Las etiquetas de clase son numeros naturales empezando en 1 hasta n, siendo n, la cantidad de clases a considerar
        
        # El recorrido se realiza por tramos, se realizan asignaciones en pasos según la cantidad de celdas dispuestas por clase
        # Este recorrido va cambiando los valores de incialización de la matriz de zonificación según la clase o tramo  
        escalar_contador_posiciones_asignadas = 0
        for escalar_etiqueta_clase, escalar_capacidad_por_clase in enumerate(vector_celdas_por_clase):
            for escalar_pasos in range(escalar_contador_posiciones_asignadas, escalar_capacidad_por_clase + escalar_contador_posiciones_asignadas):
                vector_coordenada = matriz_lista_ordenada_indices[escalar_pasos]
                matriz_zonificacion[vector_coordenada] = escalar_etiqueta_clase + 1
            escalar_contador_posiciones_asignadas += escalar_capacidad_por_clase

        return matriz_zonificacion

    def ocupacion(self, instancia_estanteria):  # Matriz dinámica
        # Evalúa el estado de ocupacion de la instancia de estantería, se asigna el valor 1 al estado ocupado y 0 al estado libre
        # Se considera una matriz de restrcción dinámica ya que cambia en función de si las posiciones estan libres o ocupadas
        
        # Se consultan las dimensiones de la estantería
        escalar_cantidad_filas, escalar_cantidad_columnas, escalar_cantidad_capas = instancia_estanteria.matriz_contenedor_items.shape
        
        # Se extrae la matriz contenedora de ítems, esta representa de forma virtual el concepto físico de estanería
        matriz_contenedor_items = instancia_estanteria.matriz_contenedor_items
        
        # Se inicializa un matriz de ceros, del mismo tamaño de la estantería
        matriz_ocupacion = np.zeros_like(matriz_contenedor_items, dtype=bool)

        # Se recorre la matriz contenedora de ítems y se evalúa si en las celdas hay instancias de ítems
        # En caso verdadero(sí hay instancia de ítem) se actualiza la matriz de restricción de 0(False) a 1(True) indicando que la celda esta vacía
        for escalar_indice_filas in range(escalar_cantidad_filas):
            for escalar_indice_columnas in range(escalar_cantidad_columnas):
                for escalar_indice_capas in range(escalar_cantidad_capas):
                    vector_indice_celda = escalar_indice_filas, escalar_indice_columnas, escalar_indice_capas
                    if isinstance(matriz_contenedor_items[vector_indice_celda], Item):
                        # indica sí está ocupado
                        matriz_ocupacion[vector_indice_celda] = 1
                    else:
                        # indica no está ocupado
                        matriz_ocupacion[vector_indice_celda] = 0

        return matriz_ocupacion


class ControladorGrua:
    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, vector_secuencia_operaciones,
                 vector_txt_configuracion_ARR, vector_capacidad_clase, txt_atributo = 'escalar_etiqueta_clase') -> None:

        # Se inicializan los atributos de clase
        self.instancia_grua = instancia_grua
        self.instancia_estanteria = instancia_estanteria
        self.vector_secuencia_operaciones = vector_secuencia_operaciones
        self.instancia_puntoES = instancia_puntoES
        self.vector_txt_configuracion_ARR = vector_txt_configuracion_ARR
        self.txt_atributo = txt_atributo
        self.instancia_matrices_priorizacion = MatricesPriorizacion(instancia_estanteria, instancia_grua, instancia_puntoES)
        self.instancia_matrices_restriccion = MatricesRestriccion()
        self.instancia_metricas = Metricas()

        # Se monta la restriccion de zonificación, usando matriz de priorización de cercanía al origen
        matriz_priorizacion_zonificacion = np.copy(self.instancia_matrices_priorizacion.devolverMatriz('Q'))
        matriz_priorizacion_zonificacion = OperacionesMatrices().desempate(matriz_priorizacion_zonificacion)
        self.matriz_restriccion_zonificacion = np.copy(self.instancia_matrices_restriccion.zonificacion(matriz_priorizacion_zonificacion, vector_capacidad_clase))

    def listarObstrucciones(self, vector_indice_celda_objetivo):
        # Esta función lista los indices de las celdas que obstruyen a la celda objetivo
        
        # Se recupera la matriz que guarda los datos sobre los ítems almacenados
        matriz_contenedor_items = self.instancia_estanteria.matriz_contenedor_items
        # Se consulta la cantidad de capas o niveles de profundidad que tiene la estantería
        escalar_cantidad_capas = matriz_contenedor_items.shape[2]

        # Se desempaquetan los indices de la celda objetivo en sus componentes
        escalar_indice_fila_obj, escalar_indice_columna_obj, escalar_indice_capa_obj = vector_indice_celda_objetivo
        
        # Se crea una lista que sevirá para almacenar los índices de las celdas que obstruyen
        vector_lista_indices_obstruyen = []

        # Se recorre desde la capa más superficial, bajando hasta llegar justamente antes de la profundidad objetivo
        # En el ciclo se evalúa si las cedas estan ocupadas por algún ítem
        # Sí se encuentran ítems en el recorrido de descenso entonces los indices se van agregando a la lista de indices que obstruyen
        for escalar_indice_capa in reversed(range(escalar_indice_capa_obj + 1, escalar_cantidad_capas)):
            if isinstance(matriz_contenedor_items[escalar_indice_fila_obj, escalar_indice_columna_obj, escalar_indice_capa], Item):
                vector_lista_indices_obstruyen.append((escalar_indice_fila_obj, escalar_indice_columna_obj, escalar_indice_capa))
                
        return vector_lista_indices_obstruyen

    def reubicar(self, vector_lista_indices_obstruyen):
        # Esta función realiza las actividades de reubicación de los ítems que obstruyen según la lista recbida como argumento
        
        # Se llaman a las variables relevantes
        matriz_zonificacion_general = np.copy(self.matriz_restriccion_zonificacion)
        instancia_grua = self.instancia_grua
        instancia_puntoES = self.instancia_puntoES
        instancia_estanteria = self.instancia_estanteria
        
        # Declaración de banderas
        booleano_irAEstanteria = False
        booleano_reubicacion_predeterminada = False

        # Se inicializan acumuladores de tiempo
        escalar_tiempo_total_reubicaciones_regulares = 0
        escalar_tiempo_total_reubicaciones_predeterminadas = 0

        # Se construye una matriz de restricciones servirá para señalizar como no factible la reubicación en la misma posición 
        # Esto para evitar la siguiente situación de ejemplo 
        # Ejemplo -> se retira el ítem que obstruye, se evalúan las posiciones libres(sí se considera matriz de priorización en tiempo la mejor posicion libre será la misma posición), se alamacena en el mismo lugar volviendose a obstruir la celda objetivo 
        matriz_no_estatico = np.ones_like(matriz_zonificacion_general)
        matriz_no_estatico[vector_lista_indices_obstruyen[0][0], vector_lista_indices_obstruyen[0][1], :] = 0

        
        # Por cada celda obstruida se realiza la reubicación respectiva
        for vector_indice_obstruye in vector_lista_indices_obstruyen:
            
            # Se consulta la matriz de ocupación de estantería
            matriz_ocupacion_general = MatricesRestriccion().ocupacion(instancia_estanteria)
            
            # Se calcula la matriz inversa logica de ocupacion
            # En esta matriz lógica las posiciones libres toman el valor 1(True)
            matriz_disponible_general = np.logical_not(matriz_ocupacion_general)

            # Se consulta el ítem que obstruye y a que clase pertence
            instancia_item_bloquea = self.instancia_estanteria.matriz_contenedor_items[vector_indice_obstruye]
            escalar_etiqueta_clase = instancia_item_bloquea.escalar_etiqueta_clase
            
            # Se calcula una matriz lógica que fija como 1(True) las celdas pertenecientes a la clase del ítem que obstruye 
            matriz_zonificacion_clase = matriz_zonificacion_general == escalar_etiqueta_clase

            # Se calcula la matriz de priorización 
            matriz_priorizacion_bruta = MatricesPriorizacion(instancia_estanteria, instancia_grua, instancia_puntoES).devolverMatriz(self.vector_txt_configuracion_ARR[1])
            
            # Se restringe la matriz de priorización al operarla con las matrices de restricción previamente calculadas
            matriz_priorizacion_acotada = matriz_priorizacion_bruta * matriz_disponible_general * matriz_zonificacion_clase * matriz_no_estatico

            # Se aplica enumeración y se desempatan los ranks 1 en caso de que sea necesario
            matriz_priorizacion = OperacionesMatrices().enumerar(matriz_priorizacion_acotada)
            matriz_priorizacion = OperacionesMatrices().desempate(matriz_priorizacion_acotada)
        
            # Se consulta en que indice esta el rank 1
            vector_indice_reubicacion = np.where(matriz_priorizacion == 1)
            vector_indice_reubicacion = tuple(np.squeeze(np.array(vector_indice_reubicacion).T))

            # Se evalúa si hay si hay alguna ubicación factible para reubicar
            if len(vector_indice_reubicacion):
                
                # en caso verdaro se evalua sí la grúa ya está en la estantería
                if not booleano_irAEstanteria:
                    # Si aún no está en la estantería entonces va a la estantería
                    escalar_tiempo_total_reubicaciones_regulares += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                    booleano_irAEstanteria = True

                # Se ejecuta la rutina de reubicación regular(reubicación dentro de estantería)
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.moverEnEstanteria(vector_indice_obstruye)
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.cargarItemDeEstanteria(vector_indice_obstruye)
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.volverAplano()
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.moverEnEstanteria(vector_indice_reubicacion)
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.soltarItemEnEstanteria(vector_indice_reubicacion)
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.volverAplano()

            else:  # Ejecutar reubicacion predeterminada
                
                # Se activa la bandera indicando que se realizo reubicación predeterminada
                booleano_reubicacion_predeterminada = True
                
                # se evalua sí la grúa ya está en la estantería
                if not booleano_irAEstanteria:
                    # Si aún no está en la estantería entonces va a la estantería
                    escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                    booleano_irAEstanteria = True

                # Se ejecuta la rutina de reubicación predeterminada(reubicación en el puntoES)
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.moverEnEstanteria(vector_indice_obstruye)
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.cargarItemDeEstanteria(vector_indice_obstruye)
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.volverAplano()
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.irAPuntoES(instancia_puntoES)
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.soltarItemEnPuntoES()
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.volverAplano()
                
                # La grúa quedo fuera de la estantería entonces se desactiva la bandera
                booleano_irAEstanteria = False

        # Finalizado el proceso de reubicación, se dejá la grúa en la esquina de la estantería
        escalar_tiempo_total_reubicaciones_regulares += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

        # Se registran las métricas
        self.instancia_metricas.registrarReubicacion(escalar_tiempo_total_reubicaciones_regulares + escalar_tiempo_total_reubicaciones_predeterminadas)

        return booleano_reubicacion_predeterminada, escalar_tiempo_total_reubicaciones_regulares, escalar_tiempo_total_reubicaciones_predeterminadas

    def almacenar(self, vector_indice_almacenamiento):
        # Esta función realiza las actividades de almacenamiento, se almacena el ítem en el índice que toma como argumento
        
        # Se llaman a las variables relevantes
        matriz_zonificacion_general = np.copy(self.matriz_restriccion_zonificacion)
        instancia_grua = self.instancia_grua
        instancia_estanteria = self.instancia_estanteria
        instancia_puntoES = self.instancia_puntoES

        # Se declara una bandera que representa el exito de tarea de almacenamiento
        booleano_item_almacenado = False

        # Se inicializan acumuladores de tiempo
        escalar_tiempo_almacenamiento = 0
        escalar_tiempo_total_reubicaciones_regulares_almacenamiento = 0
        escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento = 0

        # Se consultan los indices que obstruyen
        vector_lista_indices_obstruyen = self.listarObstrucciones(vector_indice_almacenamiento)

        # Se evalúa si hay celdas que obstruyen
        if len(vector_lista_indices_obstruyen):
            # En caso de verdadero se realiza la reubicación con el método 'reubicar'
            booleano_reubicacion_predeterminada, escalar_tiempo_total_reubicaciones_regulares_almacenamiento, escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento = self.reubicar(vector_lista_indices_obstruyen)

            # Se evalúa si se realizó reubicación predeterminada
            if booleano_reubicacion_predeterminada:
                
                # En este punto el ítem ya se encuentra despejado, entonces se realiza la rutina de almacenamiento
                escalar_tiempo_almacenamiento += instancia_grua.irAPuntoES(instancia_puntoES)
                escalar_tiempo_almacenamiento += instancia_grua.cargarItemPuntoES(indice=0)
                escalar_tiempo_almacenamiento += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                escalar_tiempo_almacenamiento += instancia_grua.moverEnEstanteria(vector_indice_almacenamiento)
                escalar_tiempo_almacenamiento += instancia_grua.soltarItemEnEstanteria(vector_indice_almacenamiento)
                escalar_tiempo_almacenamiento += instancia_grua.volverAplano()
                escalar_tiempo_almacenamiento += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

                # Se activa la bandera por el exito de la tarea
                booleano_item_almacenado = True
                
                # Luego, los ítems que reponsan temporalmente en el puntoES deben ser retornados a la estantería
                for escalar_indice_item in range(len(instancia_puntoES.matriz_contenedor_temporal_items)):
                    
                    # Se consulta la matriz de ocupación de estantería
                    matriz_ocupacion_general = MatricesRestriccion().ocupacion(instancia_estanteria)
                    
                    # Se calcula la matriz inversa lógica de ocupación
                    # En esta matriz lógica las posiciones libres toman el valor 1(True)
                    matriz_disponible_general = np.logical_not(matriz_ocupacion_general)
                    
                    # Se recupera primero el último ítem en ser almacenado en el puntoES, y su clase 
                    instancia_item_retornar = instancia_puntoES.matriz_contenedor_temporal_items[-1]
                    escalar_etiqueta_clase = instancia_item_retornar.escalar_etiqueta_clase
                    
                    # Se calcula una matriz lógica que fija como 1(True) las celdas pertenecientes a la clase del ítem que obstruye 
                    matriz_zonificacion_clase = matriz_zonificacion_general == escalar_etiqueta_clase

                    # Se calcula la matriz de priorización 
                    matriz_priorizacion_bruta = MatricesPriorizacion(instancia_estanteria, instancia_grua, instancia_puntoES).devolverMatriz(self.vector_txt_configuracion_ARR[1])
                    
                    # Se restringe la matriz de priorización al operarla con las matrices de restricción previamente calculadas
                    matriz_priorizacion_acotada = matriz_priorizacion_bruta * matriz_disponible_general * matriz_zonificacion_clase 

                    # Se aplica enumeración y se desempatan los ranks 1 en caso de que sea necesario
                    matriz_priorizacion = OperacionesMatrices().enumerar(matriz_priorizacion_acotada)
                    matriz_priorizacion = OperacionesMatrices().desempate(matriz_priorizacion_acotada)
                    
                    # Se consulta en que indice está el rank 1
                    vector_indice_retorno = np.where(matriz_priorizacion == 1)
                    vector_indice_retorno = tuple(np.squeeze(np.array(vector_indice_retorno).T))
                    
                   # Se realiza el retorno de ítem a la estantería     
                    escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento += instancia_grua.irAPuntoES(instancia_puntoES)
                    escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento += instancia_grua.cargarItemPuntoES()
                    escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                    escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento += instancia_grua.moverEnEstanteria(vector_indice_retorno)
                    escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento += instancia_grua.soltarItemEnEstanteria(vector_indice_retorno)
                    escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento += instancia_grua.volverAplano()
                    escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

        # Luego de realizar todas las reubicaciones si las hubiere, se evalúa si el ítem objetivo aún no ha sido almacenado
        if not booleano_item_almacenado:
            # en caso de que aún no se haya almacenado, entonces se realiza la rutina de almacenamiento
            escalar_tiempo_almacenamiento += instancia_grua.irAPuntoES(instancia_puntoES)
            escalar_tiempo_almacenamiento += instancia_grua.cargarItemPuntoES()
            escalar_tiempo_almacenamiento += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
            escalar_tiempo_almacenamiento += instancia_grua.moverEnEstanteria(tuple(vector_indice_almacenamiento))
            escalar_tiempo_almacenamiento += instancia_grua.soltarItemEnEstanteria(tuple(vector_indice_almacenamiento))
            escalar_tiempo_almacenamiento += instancia_grua.volverAplano()
            escalar_tiempo_almacenamiento += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

        # Finalizado el la tarea de almacenamiento, se procede a dejar la grúa en su posición de reposo
        escalar_tiempo_almacenamiento += instancia_grua.volverAReposo()

        # Se registran las métricas en el caso de que los eventos relacionadas se hayan ejecutado
        self.instancia_metricas.registrarAlmacenamiento(escalar_tiempo_almacenamiento)
        if escalar_tiempo_total_reubicaciones_regulares_almacenamiento:
            self.instancia_metricas.registrarReubicacionRegularAlmacenamiento(escalar_tiempo_total_reubicaciones_regulares_almacenamiento)
        if escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento:
            self.instancia_metricas.registrarReubicacionPredeterminadaAlmacenamiento(escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento)
        if escalar_tiempo_total_reubicaciones_regulares_almacenamiento or escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento:
            self.instancia_metricas.registrarReubicacionAlmacenamiento(escalar_tiempo_total_reubicaciones_regulares_almacenamiento + escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento)

    def recuperar(self, vector_indice_recuperacion):
        # Esta función se encarga de procesar y mandar la tarea de recuperación
        
        # Se llaman a las variables relevantes
        matriz_zonificacion_general = np.copy(self.matriz_restriccion_zonificacion)
        instancia_grua = self.instancia_grua
        instancia_estanteria = self.instancia_estanteria
        instancia_puntoES = self.instancia_puntoES

        # Se inicializa una bandera para llevar el control del exito de tarea
        booleano_item_recuperado = False

        # Se incializan acumuladores de tiempo
        escalar_tiempo_recuperacion = 0
        escalar_tiempo_total_reubicaciones_regulares_recuperacion = 0
        escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion = 0

        # Se listan obstrupciones sobre la celda objetivo
        vector_lista_indices_obstruyen = self.listarObstrucciones(vector_indice_recuperacion)

        # Se evalúa si hay celdas que obstruyen
        if len(vector_lista_indices_obstruyen):
            # En caso de verdadero se realiza la reubicación con el método 'reubicar'
            booleano_reubicacion_predeterminada, escalar_tiempo_total_reubicaciones_regulares_recuperacion, escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion = self.reubicar(vector_lista_indices_obstruyen)

            # Se evalúa si se realizó reubicación predeterminada
            if booleano_reubicacion_predeterminada:
                
                # En este punto el ítem ya se encuentra despejado, entonces se realiza la rutina de recuperación
                escalar_tiempo_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                escalar_tiempo_recuperacion += instancia_grua.moverEnEstanteria(vector_indice_recuperacion)
                escalar_tiempo_recuperacion += instancia_grua.cargarItemDeEstanteria(vector_indice_recuperacion)
                escalar_tiempo_recuperacion += instancia_grua.volverAplano()
                escalar_tiempo_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                escalar_tiempo_recuperacion += instancia_grua.irAPuntoES(instancia_puntoES)
                escalar_tiempo_recuperacion += instancia_grua.soltarItemEnPuntoESSalida()
                escalar_tiempo_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                
                # Se activa la bandera por el exito de la tarea
                booleano_item_recuperado = True

                # Luego, los ítems que reponsan temporalmente en el puntoES deben ser retornados a la estantería
                for escalar_indice_item in range(len(instancia_puntoES.matriz_contenedor_temporal_items)):
                    
                    # Se consulta la matriz de ocupación de estantería
                    matriz_ocupacion_general = MatricesRestriccion().ocupacion(instancia_estanteria)
                    
                    # Se calcula la matriz inversa lógica de ocupación
                    # En esta matriz lógica las posiciones libres toman el valor 1(True)
                    matriz_disponible_general = np.logical_not(matriz_ocupacion_general)
                    
                    # Se recupera primero el último ítem en ser almacenado en el puntoES, y su clase 
                    instancia_item_retornar = instancia_puntoES.matriz_contenedor_temporal_items[-1]
                    escalar_etiqueta_clase = instancia_item_retornar.escalar_etiqueta_clase
                    
                    # Se calcula una matriz lógica que fija como 1(True) las celdas pertenecientes a la clase del ítem que obstruye 
                    matriz_zonificacion_clase = matriz_zonificacion_general == escalar_etiqueta_clase

                    # Se calcula la matriz de priorización 
                    matriz_priorizacion_bruta = MatricesPriorizacion(instancia_estanteria, instancia_grua, instancia_puntoES).devolverMatriz(self.vector_txt_configuracion_ARR[1])
                    
                    # Se restringe la matriz de priorización al operarla con las matrices de restricción previamente calculadas
                    matriz_priorizacion_acotada = matriz_priorizacion_bruta * matriz_disponible_general * matriz_zonificacion_clase 

                    # Se aplica enumeración y se desempatan los ranks 1 en caso de que sea necesario
                    matriz_priorizacion = OperacionesMatrices().enumerar(matriz_priorizacion_acotada)
                    matriz_priorizacion = OperacionesMatrices().desempate(matriz_priorizacion_acotada)
                    
                    # Se consulta en que indice está el rank 1
                    vector_indice_retorno = np.where(matriz_priorizacion == 1)
                    vector_indice_retorno = tuple(np.squeeze(np.array(vector_indice_retorno).T))

                    # Se realiza el retorno de ítem a la estantería     
                    escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion += instancia_grua.irAPuntoES(instancia_puntoES)
                    escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion += instancia_grua.cargarItemPuntoES()
                    escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                    escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion += instancia_grua.moverEnEstanteria(vector_indice_retorno)
                    escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion += instancia_grua.soltarItemEnEstanteria(vector_indice_retorno)
                    escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion += instancia_grua.volverAplano()
                    escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

        # Luego de realizar todas las reubicaciones si las hubiere, se evalúa si el ítem objetivo aún no ha sido recuperado
        if not booleano_item_recuperado:
            # en caso de que aún no se haya almacenado, entonces se realiza la rutina de recuperación
            escalar_tiempo_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
            escalar_tiempo_recuperacion += instancia_grua.moverEnEstanteria(vector_indice_recuperacion)
            escalar_tiempo_recuperacion += instancia_grua.cargarItemDeEstanteria(vector_indice_recuperacion)
            escalar_tiempo_recuperacion += instancia_grua.volverAplano()
            escalar_tiempo_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
            escalar_tiempo_recuperacion += instancia_grua.irAPuntoES(instancia_puntoES)
            escalar_tiempo_recuperacion += instancia_grua.soltarItemEnPuntoESSalida()
            escalar_tiempo_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

        # Finalizda la actividad de recuperación se da la orden a la grúa de que regrese a reposo
        escalar_tiempo_recuperacion += instancia_grua.volverAReposo()

        # Se registran las métricas en el caso de que los eventos relacionadas se hayan ejecutado
        self.instancia_metricas.registrarRecuperacion(escalar_tiempo_recuperacion)
        if escalar_tiempo_total_reubicaciones_regulares_recuperacion:
            self.instancia_metricas.registrarReubicacionRegularRecuperacion(escalar_tiempo_total_reubicaciones_regulares_recuperacion)
        if escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion:
            self.instancia_metricas.registrarReubicacionPredeterminadaRecuperacion(escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion)
        if escalar_tiempo_total_reubicaciones_regulares_recuperacion or escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion:
            self.instancia_metricas.registrarReubicacionRecuperacion(escalar_tiempo_total_reubicaciones_regulares_recuperacion + escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion)

    def ejecutarSecuencia(self):
        # Esta funcion se encarga de ejecutar administrativamente la secuencia de operaciones
        
        # Se llaman a las variables relevantes
        vector_secuencia_operaciones = self.vector_secuencia_operaciones
        instancia_grua = self.instancia_grua
        instancia_puntoES = self.instancia_puntoES
        instancia_estanteria = self.instancia_estanteria
        
        # Se establece que el almacenamiento es 0 y la recuperación es 1, esto para facilitar la lectura del código
        escalar_almacenamiento = 0
        escalar_recuperacion = 1
        
        # Se recorre la secuencia de operaciones, operación por operación
        for vector_operacion in vector_secuencia_operaciones:
  
            # Se desmpaqueta la linea de operación, que se compone de un identificador de actividad, y un identificador de la clase
            escalar_actividad, escalar_etiqueta_clase = vector_operacion

            # se evalúa si la actvidad es almacenar
            if escalar_actividad == escalar_almacenamiento:
                
                # En caso verdadero se consulta la posición facitble en donde almacenar
                matriz_ocupacion_general = MatricesRestriccion().ocupacion(instancia_estanteria)
                matriz_disponible_general = np.logical_not(matriz_ocupacion_general)
                matriz_zonificacion_general = np.copy(self.matriz_restriccion_zonificacion)
                matriz_zonificacion_clase = matriz_zonificacion_general == escalar_etiqueta_clase
                matriz_priorizacion_bruta = MatricesPriorizacion(instancia_estanteria, instancia_grua,instancia_puntoES).devolverMatriz(self.vector_txt_configuracion_ARR[0])
                matriz_priorizacion_acotada = matriz_priorizacion_bruta * matriz_disponible_general * matriz_zonificacion_clase
                
                # Se aplica enumeración y se desempatan los ranks 1 en caso de que sea necesario
                matriz_priorizacion = OperacionesMatrices().enumerar(matriz_priorizacion_acotada)
                matriz_priorizacion = OperacionesMatrices().desempate(matriz_priorizacion_acotada)

                vector_indice_almacenamiento = np.where(matriz_priorizacion == 1)
                vector_indice_almacenamiento = tuple(np.squeeze(np.array(vector_indice_almacenamiento).T))

                # Se evalúa si dicha posicion factible existe
                if len(vector_indice_almacenamiento):
                    # En caso verdadero, entonces realizar el almacenamiento del ítem en la posición establecida
                    self.instancia_puntoES.matriz_contenedor_temporal_items.append(Item(escalar_etiqueta_clase))
                    self.almacenar(vector_indice_almacenamiento)
                else:
                    # En caso de que no exista posición factible, de forma predeterminada se descarta la operación de almacenamiento
                    # Este caso representa que no hay espacio disponible para almacenar el ítem 
                    pass
                
            # O se evalúa si es recuperación
            elif escalar_actividad == escalar_recuperacion:
                
                # En caso verdadero se consulta si hay algun ítem disponible que coincida con la etiqueta de clase solicitada
                matriz_ocupacion_general = MatricesRestriccion().ocupacion(instancia_estanteria)
                matriz_zonificacion_general = np.copy(self.matriz_restriccion_zonificacion)
                matriz_zonificacion_clase = matriz_zonificacion_general == escalar_etiqueta_clase
                matriz_ocupacion_clase = matriz_ocupacion_general * matriz_zonificacion_clase
                matriz_priorizacion_bruta = MatricesPriorizacion(instancia_estanteria, instancia_grua, instancia_puntoES).devolverMatriz(self.vector_txt_configuracion_ARR[2])
                matriz_priorizacion_acotada = matriz_priorizacion_bruta * matriz_ocupacion_clase
                
                # Se aplica enumeración y se desempatan los ranks 1 en caso de que sea necesario
                matriz_priorizacion = OperacionesMatrices().enumerar(matriz_priorizacion_acotada)
                matriz_priorizacion = OperacionesMatrices().desempate(matriz_priorizacion_acotada)

                vector_indice_recuperacion = np.where(matriz_priorizacion == 1)
                vector_indice_recuperacion = tuple(np.squeeze(np.array(vector_indice_recuperacion).T))

                # Se evalúa si efectivamente se encontro alguna celda que contenga el ítem solicitado
                if len(vector_indice_recuperacion):
                    
                    # En caso verdadero, se realiza la recuperación del ítem
                    self.recuperar(vector_indice_recuperacion)

                else:
                    # En caso de no exista el ítem,  entonces se omite la operación de recuperación
                    pass
            
class Simulacion:
    def __init__(self):
        # Se contruye la instancia de simulación con la configuración fijada
        self.config = self.cargarConfiguracion()

    def cargarConfiguracion(self):
        # Se cargan los párametros del modelo
        config = {
            
            # Nota en los párametros espaciales las componentes verticales y horizontales son intercambiables, solo se debe tener congruencia en todos los demás párametros
            
            # Se recomienda utilizar la convención de ejemplo, esto por que la matrices tradicionalmente se recorrecoren por filas y luego columnas
            # Sin embargo como se meciona, siempre y cuando se conserve congruencia las componente  verticales y horizontales son intercambiales sin implicaciones negativas
            
            "vector_posicion_estanteria": np.array([0, 0, 0]), # (posición vertical, posición horizontal, posición de profundidad)
            "vector_configuracion_estanteria": np.array([18, 20, 2]), # (filas, columnas, capas de profundidad)
            "vector_cotas_estanteria": np.array([1.6, 0.9, 1.1]), # (alto de fila, ancho de columna, profundid de celda)
            "vector_centro_celda_estanteria": np.array([0, 0, 0]), # (% verical, % horizontal, % profundidad), representa un traslación del punto de referencia dentro de la celda
            "escalar_nivel_ocupacion": 0.85, # Porcentaje de la estatería que esta ocupada, intervalo [0, 1]
            "escalar_separacion_riel_grua": 0.5, # Unidades de separación entre la cara frontal de la estantería y el riel de la grúa
            "vector_velocidad_grua": np.array([1, 1.5, 0.7]), #(Velocidad en filas, velocidad en columnas, velocidad a lo largo de canales)
            "escalar_tiempo_carga_descarga": 0, # Tiempo que demora la grúa en cargar o descargar algún ítem
            "vector_capacidad_clase": np.array([0.2, 0.4, 0.4]), # Vector de proporción que describe los porcentajes correspondientes a las cantidades a asignar a cada clase
            "escalar_numero_cantidad_clases": 3,  # Este valor debe coincider con la cantidad de elementos contenidos en el parámetro anterior
            "vector_prob_acum_AR": np.array([0.5, 1]), # Vector de probabilidad acumulada, para el almacenamiento y recuperación
            "escalar_largo_secuencia": 1,  # Valor inicial, se ajustará en la inicialización
            "escalar_cantidad_replicas": 1000, # la simulación se repetira según la cantidad de replicas
            "vector_politicas_almacenamiento": ('P'), # Se establecen las políticas de almacenamiento a considerar
            "vector_politicas_reubicacion": ('P','K', 'Q', 'G', 'X'), # Se establecen las politicas de reubicación a considerar
            "vector_politicas_recuperacion": ('A',) # Se establecen las politicas de recuperación a considerar
        }
        # Se calcula el largo de secuencia
        config["escalar_largo_secuencia"] = round(np.prod(config["vector_configuracion_estanteria"]) * 0.1)
        
        return config

    def inicializarSimulacion(self):
        # Se carga el diccionario de configuración de la simulación
        cfg = self.config
        
        # De forma intuitiva se establece una separación entre el riel y la cara frontal de la estantería
        vector_coordenada_z_riel_grua = cfg["escalar_separacion_riel_grua"] + (cfg["vector_configuracion_estanteria"][2] * cfg["vector_cotas_estanteria"][2])
        
        # A fin de simplificar el modelo se establece que el punto de referencia central o centro del espacio es el punto ES
        vector_posicion_puntoES = np.array([0, 0, vector_coordenada_z_riel_grua])
        
        # A fin de simplificar el modelo se establece que el punto de reposo de la grúa es la misma que la del puntoES
        vector_posicion_reposo_grua = np.array([0, 0, vector_coordenada_z_riel_grua])
        
        # Se calculan las probabilidades acumuladas de Hausman
        vector_prob_acum_A_clasificada = _GeneradorProbAcumuladas().hausman(cfg["escalar_numero_cantidad_clases"], 1)
        vector_prob_acum_R_clasificada = _GeneradorProbAcumuladas().hausman(cfg["escalar_numero_cantidad_clases"], 1)

        # Se inicializan las intancias de los objetos
        E = Estanteria(cfg["vector_posicion_estanteria"], cfg["vector_configuracion_estanteria"], cfg["vector_cotas_estanteria"], cfg["vector_centro_celda_estanteria"])
        G = Grua(vector_posicion_reposo_grua, cfg["vector_velocidad_grua"], cfg["escalar_tiempo_carga_descarga"])
        P = PuntoES(vector_posicion_puntoES)
        
        
        # Se cálcaula la cantidad de almacenemiento a realizar en la inicialización, con el fin de alzcanzar el nivel de ocupación fijado
        escalar_largo_secuencia_fijado = int(E.escalar_capacidad_total * cfg["escalar_nivel_ocupacion"])
        
        # Se genra la seucuencia de llenado
        SA = GeneradorSecuencia(escalar_largo_secuencia_fijado, np.array([1, 0]), vector_prob_acum_A_clasificada, vector_prob_acum_R_clasificada).generarSecuencia()
        
        # Se llama una instancia de ControladoGrua, se cargan los parámetros de inicialización
        CA = ControladorGrua(E, G, P, SA, ('Q', 'X', 'X'), cfg["vector_capacidad_clase"])
        
        # Se ejecuta la secuencia de llenado al nivel de ocupación fijado
        CA.ejecutarSecuencia()
        
        # Se genera la secuencia de operación a ejecutar luego de la inicialización
        S = GeneradorSecuencia(cfg["escalar_largo_secuencia"], cfg["vector_prob_acum_AR"], vector_prob_acum_A_clasificada, vector_prob_acum_R_clasificada).generarSecuencia()

        return E, G, P, S

    def prepararEscenarios(self, vector_instancias_EGPS):
        # Se carga el diccionario de configuración de la simulación
        cfg = self.config
        
        # Se cruzan las politicas de almacenamiento, reubicación y recuperación, esto con el fin de obtener los escenarios de politicas conjudas a evaluar
        matriz_lista_politicas_conjugadas = list(product(cfg["vector_politicas_almacenamiento"], cfg["vector_politicas_reubicacion"], cfg["vector_politicas_recuperacion"]))
        
        # Se crea una lista en donde se guardaran elementos de cada escenario a evaluar
        matriz_lista_rutina_escenarios = []

        # Se realiza una copia de las instancias EGPS incializadas, po cada politica conjugada, y s su vez según la cantidad de replicas,
        # cada eslabón se integra en un vector que servirá para ser pasadado como argumento de ControladorGrua
        # Todos los arreglos de argumentos son almacenados en la lista de rutina de escenarios
        
        for _ in range(cfg["escalar_cantidad_replicas"]):
            for vector_politica_conjugada in matriz_lista_politicas_conjugadas:
                copia_vector_instancias_EGPS = copy.deepcopy(vector_instancias_EGPS)
                vector_argumentos_ControladorGrua = (*copia_vector_instancias_EGPS, vector_politica_conjugada, cfg["vector_capacidad_clase"])
                matriz_lista_rutina_escenarios.append(vector_argumentos_ControladorGrua)
        
        return matriz_lista_rutina_escenarios
    
    @staticmethod
    def cargarEscenario(vector_argumentos_ControladorGrua):
        # Se crea un metodo auxilizar que servirá para procesar de forma dedicada un vector de argumentos y arrojar un vector de métricas  
        C = ControladorGrua(*vector_argumentos_ControladorGrua)
        C.ejecutarSecuencia()
        
        # Se consulta la política conjugada
        vector_politica_conjugada = vector_argumentos_ControladorGrua[4]
        
        # Se recuperan las métricas resultado de la simulación del eslabón
        vector_resultados_escenario = C.instancia_metricas.recuperarInformcion()
        
        # Con el fin de llevar trazabilidad, incluye en el vector de resultados la politica utilizada
        vector_resultados_escenario.insert(0, ''.join(vector_politica_conjugada))
        return vector_resultados_escenario

    
    def ejecutarSimulacion(self):
        
        # en esta función se ingran las funciones anteriores
        # Se ejecuta la simulación usando procesamiento en paralelo a fin de reducir el tiempo de computo,
        
        # Se incicializan las instancias EGPS
        vector_instancias_EGPS = self.inicializarSimulacion()
        
        # Se obtiene la lista los escenarios
        matriz_lista_rutina_escenarios = self.prepararEscenarios(vector_instancias_EGPS)

        # Se crea una bahía de procesos 
        # Si se deja 'processes=mp.cpu_count()' entonces usará todos los núcleos
        # En cambio si se remplaza 'mo.cpu_count()' por algún número entero menor que la cantidad de núcleos disponibles en el procesador, entonces se limitara la cantidad de núcleos utilizados
        with mp.Pool(processes=mp.cpu_count()) as bahia:
            resultados = bahia.map(Simulacion.cargarEscenario, matriz_lista_rutina_escenarios)

        self.guardarResultadosCSV(resultados)

    @staticmethod
    def guardarResultadosCSV(resultados):
        
        # Es una función de utilidad, ayuda a escribir los resultados de la simulación en un archivo CSV
        with open('resultados_simulacion.csv', 'w', newline='') as archivo:
            
            # Dependiendo de la configuración regional del lector de hojas de cálculo en ciertas ocaciones el .csv no se visualiza correctamente
            # Entonces, se recomienda modificar el delimitador/delimiter según sea necesario
            writer = csv.writer(archivo, delimiter=';')
            writer.writerow(["Pol_Control", 
                                  "Cant_Almacenamientos", "TiempoTotal_Almacenamiento", "DesvEst_Tiempo_Almacenamiento", #A
                                  "Cant_Recuperaciones", "TiempoTotal_Recuperacion", "DesvEst_Tiempo_Recuperacion", #B
                                  "Cant_Reubicaciones", "TiempoTotal_Reubicacion", "DesvEst_Tiempo_Reubicacion", #C
                                  "Cant_ReubicacionesAlmacenamiento", "TiempoTotal_ReubicacionesAlmacenamiento", "DesvEst_Tiempo_ReubicacionesAlmacenamiento", #D
                                  "Cant_ReubicacionesRecuperacion", "TiempoTotal_ReubicacionesRecuperacion", "DesvEst_Tiempo_ReubicacionesRecuperacion", #E
                                  "Cant_ReubicacionesRegularesAlmacenamiento", "TiempoTotal_ReubicacionesRegularesAlmacenamiento", "DesvEst_Tiempo_ReubicacionesRegularesAlmacenamiento", #F
                                  "Cant_ReubicacionesPredeterminadasAlmacenamiento", "TiempoTotal_ReubicacionesPredeterminadasAlmacenamiento", "DesvEst_Tiempo_ReubicacionesPredeterminadasAlmacenamiento", #G
                                  "Cant_ReubicacionesRegularesRecuperacion", "TiempoTotal_ReubicacionesRegularesRecuperacion", "DesvEst_Tiempo_ReubicacionesRegularesRecuperacion", #H
                                  "Cant_ReubicacionesPredeterminadasRecuperacion", "TiempoTotal_ReubicacionesPredeterminadasRecuperacion", "DesvEst_Tiempo_ReubicacionesPredeterminadasRecuperacion", #I
                                  "Cant_Operaciones", "TiempoTotal_Operacion"]) 
            writer.writerows(resultados)
            
if __name__ == "__main__":
    escalar_tiempo_inicio = time.time()
    Simulacion().ejecutarSimulacion()
    escalar_tiempo_fin = time.time()
    escalar_duracion = (escalar_tiempo_fin - escalar_tiempo_inicio) / 60
    
    print(f'La simulación fue efecutada en {escalar_duracion} minutos.')
    
     
