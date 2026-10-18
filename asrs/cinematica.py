"""Cinemática de la grúa: tiempo de recorrido de cada eje con aceleración y deceleración."""
import math

import numpy as np


def tiempo_recorrido(distancia, velocidad, aceleracion=0, deceleracion=0):
    # Tiempo que tarda un eje en recorrer una distancia con perfil de velocidad trapezoidal
    #   Acelera hasta la velocidad máxima, avanza a velocidad constante y frena.
    #   Si la distancia es corta no alcanza la velocidad máxima y el perfil es triangular.
    # Una aceleración (o deceleración) de 0 significa sin límite: esa fase es instantánea.
    # Con las dos en 0 el tiempo es distancia / velocidad (velocidad constante).
    # Admite escalares o arreglos que se combinan por broadcasting
    distancia, velocidad, aceleracion, deceleracion = np.broadcast_arrays(
        np.abs(np.asarray(distancia, dtype=float)), np.asarray(velocidad, dtype=float),
        np.asarray(aceleracion, dtype=float), np.asarray(deceleracion, dtype=float))

    # Inversas de las aceleraciones, 0 cuando no hay límite
    with np.errstate(divide='ignore', invalid='ignore'):
        inversa_a = np.where(aceleracion > 0, 1.0 / aceleracion, 0.0)
        inversa_d = np.where(deceleracion > 0, 1.0 / deceleracion, 0.0)

    # Distancia necesaria para alcanzar la velocidad máxima y para frenar desde ella
    distancia_acelerando = velocidad ** 2 * inversa_a / 2
    distancia_frenando = velocidad ** 2 * inversa_d / 2
    distancia_minima = distancia_acelerando + distancia_frenando

    # Perfil trapezoidal
    tiempo_trapezoidal = velocidad * (inversa_a + inversa_d) + (distancia - distancia_minima) / velocidad

    # Perfil triangular: la velocidad pico es menor que la máxima
    with np.errstate(divide='ignore', invalid='ignore'):
        velocidad_pico = np.sqrt(2 * distancia / (inversa_a + inversa_d))
        tiempo_triangular = velocidad_pico * (inversa_a + inversa_d)

    return np.where(distancia >= distancia_minima, tiempo_trapezoidal, tiempo_triangular)


def tiempo_chebyshev_xy(distancias_xy, velocidad_xy, aceleracion_xy=0, deceleracion_xy=0):
    # Tiempo del tramo xy: los dos ejes se mueven a la vez, cada uno con su propio perfil, y el tramo termina cuando llega el más lento
    # distancias_xy: arreglo cuya última dimensión es 2 (filas, columnas)
    return np.max(tiempo_recorrido(distancias_xy, velocidad_xy, aceleracion_xy, deceleracion_xy), axis=-1)


def tiempo_recorrido_escalar(distancia, velocidad, aceleracion=0.0, deceleracion=0.0):
    # Mismo cálculo que tiempo_recorrido para un solo eje con números de Python
    # Es mucho más rápido que la versión con arreglos y es la que usa la grúa, que calcula miles de desplazamientos
    distancia = abs(distancia)
    inversa_a = 1.0 / aceleracion if aceleracion > 0 else 0.0
    inversa_d = 1.0 / deceleracion if deceleracion > 0 else 0.0
    suma = inversa_a + inversa_d
    distancia_minima = velocidad * velocidad * suma / 2
    if distancia >= distancia_minima:
        return velocidad * suma + (distancia - distancia_minima) / velocidad
    return math.sqrt(2 * distancia / suma) * suma
