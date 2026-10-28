"""Perfil de cada SKU: lo que el agente observa de su demanda pasada, sin identidad ni curvas de demanda."""
import numpy as np

EPSILON_TASA = 0.1 # Recuperaciones por día que se suman a cada tasa para que el logaritmo de la tendencia exista: regularización numérica, no un parámetro del modelo


def historia_en_dias(escenario, escalar_dia, escalar_ventana):
    # Días desde la primera llegada de cada SKU, acotados a la ventana de popularidad
    # Un SKU que ya tenía existencias el día 0 de la prehistoria tiene la historia completa; uno que aún no ha entrado tiene 0
    vector_primera_llegada = np.asarray(escenario.vector_dia_primera_llegada, dtype=float)
    vector_dias = np.where(np.isfinite(vector_primera_llegada), escalar_dia - vector_primera_llegada, 0.0)
    return np.clip(vector_dias, 0.0, escalar_ventana)


def perfil_de_skus(escenario, escalar_dia, escalar_ventana=30, escalar_duracion_ciclo=None):
    # Perfil de cada SKU el día de operación dado, con una fila por SKU (indexada por SKU - 1) y las columnas:
    #   0  nivel: ln(1 + r), con r la tasa de recuperaciones por día dentro de su historia (recuperaciones de la ventana / días con historia)
    #   1  tendencia: ln((r reciente + ε) / (r previa + ε)), con las tasas de la mitad de la ventana más reciente y de la anterior; 0 si aún no hay una mitad previa
    #   2  historia: días desde la primera llegada, acotados a la ventana, divididos entre la ventana (0 si es nuevo, 1 si ya está establecido)
    #   3  edad (solo si se da la duración del ciclo): edad real del producto dividida entre la duración, la variable extra de la ablación
    # Solo usa recuperaciones de días anteriores al dado, es decir, lo que el almacén ya observó. Las filas de los SKU que no existen no tienen significado
    escalar_mitad = escalar_ventana // 2
    vector_historia = historia_en_dias(escenario, escalar_dia, escalar_ventana)
    vector_total = escenario.recuperaciones(escalar_dia - escalar_ventana, escalar_dia).sum(axis=0)
    vector_reciente = escenario.recuperaciones(escalar_dia - escalar_mitad, escalar_dia).sum(axis=0)
    vector_previa = escenario.recuperaciones(escalar_dia - 2 * escalar_mitad, escalar_dia - escalar_mitad).sum(axis=0)

    vector_tasa = vector_total / np.maximum(1.0, vector_historia)
    vector_tasa_reciente = vector_reciente / np.maximum(1.0, np.minimum(escalar_mitad, vector_historia))
    vector_tasa_previa = vector_previa / np.maximum(1.0, np.minimum(escalar_mitad, vector_historia - escalar_mitad))
    vector_tendencia = np.where(vector_historia > escalar_mitad,
                                np.log((vector_tasa_reciente + EPSILON_TASA) / (vector_tasa_previa + EPSILON_TASA)), 0.0)

    lista_columnas = [np.log1p(vector_tasa), vector_tendencia, vector_historia / escalar_ventana]
    if escalar_duracion_ciclo is not None:
        lista_columnas.append(np.clip((escalar_dia - np.asarray(escenario.vector_dia_lanzamiento, dtype=float)) / escalar_duracion_ciclo, 0.0, 1.0))
    return np.column_stack(lista_columnas)
