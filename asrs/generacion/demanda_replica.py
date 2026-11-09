"""Demanda propia de cada réplica (capítulo 8, secciones 8.2.3 y 8.2.4).

Cada réplica obtiene su secuencia con elementos que fija su semilla: el muestreo ponderado del surtido y el punto de inicio (rotación circular),
cuando la demanda sale de una fuente, y la mezcla diaria de la Dirichlet, cuando el escenario tiene un nivel de variabilidad.
"""
import os
from functools import lru_cache

import numpy as np

from asrs.generacion.fuentes.base import extraer_secuencia, leer_lineas
from asrs.generacion.fuentes.registro import ADAPTADORES, crear_adaptador
from asrs.generacion.variabilidad import estimar_omega, mezclar_dirichlet, omega_del_nivel, recuperaciones_por_dia

# Carpeta por defecto de las fuentes: tesis/datos, hermana del repositorio del simulador
CARPETA_DATOS_POR_DEFECTO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tesis", "datos"))


def ruta_fuente(txt_fuente, txt_carpeta_datos=""):
    # Ruta del archivo crudo de una fuente registrada, dentro de <carpeta>/<carpeta de la fuente>/crudo/
    if txt_fuente not in ADAPTADORES:
        raise ValueError(f"Fuente de demanda: '{txt_fuente}' no existe. Disponibles: {', '.join(ADAPTADORES)}.")
    txt_carpeta, txt_archivo, _ = ADAPTADORES[txt_fuente]
    return os.path.join(txt_carpeta_datos or CARPETA_DATOS_POR_DEFECTO, txt_carpeta, "crudo", txt_archivo)


@lru_cache(maxsize=4)
def _lineas_en_cache(txt_fuente, txt_ruta, escalar_modificado):
    # Las líneas filtradas y ordenadas de una fuente se leen una vez por proceso: el archivo crudo es lo más lento del tratamiento
    return leer_lineas(crear_adaptador(txt_fuente), txt_ruta)


def secuencia_de_fuente(txt_fuente, txt_carpeta_datos, escalar_semilla, escalar_recuperaciones, escalar_dias_modelo):
    """Trata la fuente con la semilla de la réplica: muestreo ponderado del surtido, rotación circular, calendario reescalado a D días y volumen
    escalado a N recuperaciones. Devuelve el `ResultadoExtraccion`, con el día del modelo de cada recuperación en `vector_dias`."""
    txt_ruta = ruta_fuente(txt_fuente, txt_carpeta_datos)
    if not os.path.isfile(txt_ruta):
        raise ValueError(f"Fuente de demanda '{txt_fuente}': no encuentro el archivo crudo {txt_ruta}. Los datos no se versionan; "
                         "cópialos a esa carpeta o indica otra en 'texto_carpeta_datos'.")
    lineas = _lineas_en_cache(txt_fuente, txt_ruta, os.path.getmtime(txt_ruta))
    return extraer_secuencia(crear_adaptador(txt_fuente), txt_ruta, "adaptador", escalar_semilla, True, escalar_recuperaciones,
                             escalar_dias_modelo, True, lineas=lineas)


SEMILLA_OMEGA = 0 # Semilla del tratamiento de referencia con que se estima el ω̂ de una fuente: el mismo para todas las réplicas


@lru_cache(maxsize=8)
def omega_de_fuente(txt_fuente, txt_carpeta_datos, escalar_recuperaciones, escalar_dias_modelo, escalar_ventana):
    """ω̂ de una fuente (ec. 8.8), estimado sobre su secuencia tratada y escalada con la semilla de referencia `SEMILLA_OMEGA`.
    Es un valor por fuente: todas las réplicas usan el mismo, de modo que cada nivel del factor de variabilidad es idéntico entre réplicas."""
    resultado = secuencia_de_fuente(txt_fuente, txt_carpeta_datos, SEMILLA_OMEGA, escalar_recuperaciones, escalar_dias_modelo)
    return estimar_omega(recuperaciones_por_dia(resultado.vector_skus, resultado.vector_dias)[0], escalar_ventana)


def demanda_de_replica(vector_skus, vector_dias, txt_variabilidad, escalar_ventana, escalar_semilla, escalar_omega_fuente=None):
    """Aplica el nivel de variabilidad a una secuencia tratada (SKU, día) con el ω̂ de la fuente, `escalar_omega_fuente`; si es None se estima con
    la propia secuencia (ec. 8.8). Con 'ninguna' la secuencia queda igual. El sorteo usa `numpy.random.default_rng([semilla, 2])`, un flujo
    distinto del muestreo (semilla) y de la rotación ([semilla, 1]).
    Devuelve (skus, días, resumen) con ω̂ de la fuente, el de la secuencia de la réplica (diagnóstico) y el ω aplicado."""
    escalar_omega_replica = estimar_omega(recuperaciones_por_dia(vector_skus, vector_dias)[0], escalar_ventana)
    escalar_omega_hat = escalar_omega_replica if escalar_omega_fuente is None else escalar_omega_fuente
    escalar_omega = omega_del_nivel(txt_variabilidad, escalar_omega_hat)
    dic_resumen = {"variabilidad": txt_variabilidad, "omega_fuente": round(float(escalar_omega_hat), 6),
                   "omega_replica": round(float(escalar_omega_replica), 6),
                   "omega": None if escalar_omega is None else round(float(escalar_omega), 6)}
    if escalar_omega is None:
        return list(vector_skus), list(vector_dias), dic_resumen
    vector_skus, vector_dias = mezclar_dirichlet(vector_skus, vector_dias, escalar_omega, escalar_ventana, np.random.default_rng([escalar_semilla, 2]))
    return vector_skus, vector_dias, dic_resumen
