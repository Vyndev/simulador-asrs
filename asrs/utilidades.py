"""Utilidades numéricas compartidas."""
import numpy as np


def repartir_entero(escalar_total, vector_proporciones, escalar_minimo=0):
    # Reparte un total entero según proporciones con el método del mayor resto: la suma siempre es igual al total
    # escalar_minimo: cantidad mínima que recibe cada elemento (por ejemplo, una celda por SKU)
    vector_proporciones = np.asarray(vector_proporciones, dtype=float)
    escalar_n = len(vector_proporciones)
    escalar_libres = int(escalar_total) - escalar_minimo * escalar_n
    if escalar_libres < 0:
        raise ValueError(f"No se puede repartir {escalar_total} entre {escalar_n} elementos con mínimo {escalar_minimo}.")
    escalar_suma = vector_proporciones.sum()
    if escalar_n == 0 or escalar_suma <= 0:
        raise ValueError("Las proporciones deben sumar más que cero.")

    vector_cuotas = vector_proporciones / escalar_suma * escalar_libres
    vector_base = np.floor(vector_cuotas + 1e-9).astype(int)
    escalar_faltantes = int(escalar_libres - vector_base.sum())
    if escalar_faltantes > 0:
        vector_orden = np.argsort(-(vector_cuotas - vector_base), kind='stable')
        for escalar_indice in vector_orden[:escalar_faltantes]:
            vector_base[escalar_indice] += 1
    return vector_base + escalar_minimo
