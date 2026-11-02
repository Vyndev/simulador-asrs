"""Mide qué tan bien el perfil del SKU separa un producto que sube de uno en declive, con y sin la edad real.

Uso: python scripts/separabilidad_perfil.py
Genera escenarios de un año con el generador, calcula el perfil de cada SKU establecido (historia completa) cada 10 días y lo etiqueta con la fase
real de su ciclo (creciente: edad entre 30 días y 0,4 L; en declive: edad entre 0,6 L y L). Ajusta una regresión logística con semillas de entrenamiento
y mide el acierto con semillas de prueba, para varios conjuntos de variables.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asrs.agente.perfil import perfil_de_skus
from asrs.config import cargar_configuracion
from asrs.generacion.ciclo_vida import DIAS_POR_MES
from asrs.generacion.reposicion import generar_escenario

SEMILLAS_ENTRENAMIENTO = (1, 2, 3, 4)
SEMILLAS_PRUEBA = (5, 6, 7)


def datos(escalar_semilla, escalar_duracion):
    config = dict(cargar_configuracion(), escalar_largo_secuencia=365 * 1440)
    escenario = generar_escenario(config, np.random.default_rng(escalar_semilla))
    lista_x, lista_y = [], []
    for escalar_dia in range(30, 360, 10):
        matriz_perfil = perfil_de_skus(escenario, escalar_dia, 30, escalar_duracion)
        vector_edad = (escalar_dia - escenario.vector_dia_lanzamiento) / escalar_duracion
        vector_ok = escenario.activos(escalar_dia) & (matriz_perfil[:, 2] == 1.0)
        vector_creciente = vector_ok & (vector_edad * escalar_duracion >= 30) & (vector_edad <= 0.4)
        vector_declive = vector_ok & (vector_edad >= 0.6) & (vector_edad < 1.0)
        vector_elegido = vector_creciente | vector_declive
        lista_x.append(matriz_perfil[vector_elegido])
        lista_y.append(vector_declive[vector_elegido].astype(float))
    return np.vstack(lista_x), np.concatenate(lista_y)


def ajustar_logistica(matriz_x, vector_y, escalar_iteraciones=3000, escalar_paso=0.5):
    vector_media, vector_desviacion = matriz_x.mean(axis=0), matriz_x.std(axis=0) + 1e-9
    matriz_z = np.column_stack([np.ones(len(matriz_x)), (matriz_x - vector_media) / vector_desviacion])
    vector_w = np.zeros(matriz_z.shape[1])
    for _ in range(escalar_iteraciones):
        vector_p = 1 / (1 + np.exp(-matriz_z @ vector_w))
        vector_w -= escalar_paso * matriz_z.T @ (vector_p - vector_y) / len(vector_y)
    return vector_w, vector_media, vector_desviacion


def acierto(matriz_x, vector_y, parametros):
    vector_w, vector_media, vector_desviacion = parametros
    matriz_z = np.column_stack([np.ones(len(matriz_x)), (matriz_x - vector_media) / vector_desviacion])
    return float((((matriz_z @ vector_w) > 0) == (vector_y > 0.5)).mean())


def main():
    escalar_duracion = cargar_configuracion()["escalar_meses_ciclo_vida"] * DIAS_POR_MES
    entrenamiento = [datos(s, escalar_duracion) for s in SEMILLAS_ENTRENAMIENTO]
    prueba = [datos(s, escalar_duracion) for s in SEMILLAS_PRUEBA]
    matriz_xe, vector_ye = np.vstack([x for x, _ in entrenamiento]), np.concatenate([y for _, y in entrenamiento])
    matriz_xp, vector_yp = np.vstack([x for x, _ in prueba]), np.concatenate([y for _, y in prueba])
    print(f"Observaciones: {len(vector_ye)} de entrenamiento y {len(vector_yp)} de prueba; proporción en declive {vector_yp.mean():.2f}")
    print("Acierto al separar un SKU creciente de uno en declive (con historia completa):")
    for txt_nombre, vector_columnas in (("nivel", [0]), ("nivel y tendencia", [0, 1]), ("nivel, tendencia y edad real", [0, 1, 3])):
        parametros = ajustar_logistica(matriz_xe[:, vector_columnas], vector_ye)
        print(f"  {txt_nombre:32s} {acierto(matriz_xp[:, vector_columnas], vector_yp, parametros):.3f}")
    vector_signo = (matriz_xp[:, 1] < 0).astype(float)
    print(f"  solo el signo de la tendencia    {float((vector_signo == vector_yp).mean()):.3f}")


if __name__ == "__main__":
    main()
