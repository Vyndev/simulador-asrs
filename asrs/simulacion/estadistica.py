"""Resumen estadístico de varias réplicas: promedio, dispersión y cuántas veces gana cada política."""
import math
from statistics import NormalDist

# Posición de cada métrica en una fila de resultados (la fila empieza con los nombres de la política de asignación y de recuperación)
COLUMNAS_METRICAS = {
    "tiempo_total": -1,
    "tiempo_almacenamiento": 3,
    "tiempo_recuperacion": 6,
    "tiempo_reubicacion": 9,
    "reubicaciones": 8,
    "operaciones": -2,
}


def cuantil_t_975(escalar_gl):
    # Cuantil 0,975 de la t de Student con escalar_gl grados de libertad, para el intervalo de confianza del 95 %
    # Con 1 y 2 grados de libertad la fórmula es exacta; con más se usa el desarrollo de Cornish-Fisher alrededor
    # del cuantil normal (Abramowitz y Stegun, 26.7.5), con un error menor al 0,2 %
    p = 0.975
    if escalar_gl == 1:
        return math.tan(math.pi * (p - 0.5))
    if escalar_gl == 2:
        return (2 * p - 1) * math.sqrt(2 / (4 * p * (1 - p)))
    x = NormalDist().inv_cdf(p)
    g1 = (x ** 3 + x) / 4
    g2 = (5 * x ** 5 + 16 * x ** 3 + 3 * x) / 96
    g3 = (3 * x ** 7 + 19 * x ** 5 + 17 * x ** 3 - 15 * x) / 384
    g4 = (79 * x ** 9 + 776 * x ** 7 + 1482 * x ** 5 - 1920 * x ** 3 - 945 * x) / 92160
    n = escalar_gl
    return x + g1 / n + g2 / n ** 2 + g3 / n ** 3 + g4 / n ** 4


def _estadisticos(vector_valores):
    escalar_n = len(vector_valores)
    escalar_media = sum(vector_valores) / escalar_n
    escalar_desviacion = math.sqrt(sum((v - escalar_media) ** 2 for v in vector_valores) / (escalar_n - 1)) if escalar_n > 1 else 0.0
    return {
        "media": escalar_media,
        "desviacion": escalar_desviacion,
        "minimo": min(vector_valores),
        "maximo": max(vector_valores),
        # Intervalo de confianza del 95 % de la media con la t de Student (la desviación se estima con las mismas réplicas)
        "ic95": cuantil_t_975(escalar_n - 1) * escalar_desviacion / math.sqrt(escalar_n) if escalar_n > 1 else 0.0,
    }


def resumir(resultados_por_replica):
    # resultados_por_replica: lista con una entrada por réplica; cada entrada es la lista de filas de esa réplica, una por escenario (asignación y recuperación)
    # Devuelve un diccionario con las estadísticas por política y el conteo de victorias (comparación pareada por réplica)
    escalar_replicas = len(resultados_por_replica)
    if escalar_replicas == 0:
        return {"replicas": 0, "politicas": []}

    vector_nombres = [(fila[0], fila[1]) for fila in resultados_por_replica[0]]
    vector_victorias = [0] * len(vector_nombres)
    for vector_filas in resultados_por_replica:
        vector_tiempos = [float(fila[-1]) for fila in vector_filas]
        escalar_mejor = min(vector_tiempos)
        for escalar_indice, escalar_tiempo in enumerate(vector_tiempos):
            if escalar_tiempo <= escalar_mejor + 1e-9:
                vector_victorias[escalar_indice] += 1

    vector_politicas = [{"nombre": txt_nombre, "recuperacion": txt_recuperacion, "victorias": vector_victorias[escalar_indice]}
                        for escalar_indice, (txt_nombre, txt_recuperacion) in enumerate(vector_nombres)]
    for txt_metrica, escalar_columna in COLUMNAS_METRICAS.items():
        matriz_valores = [[float(vector_filas[escalar_indice][escalar_columna]) for vector_filas in resultados_por_replica] for escalar_indice in range(len(vector_nombres))]
        vector_medias = [sum(vector_valores) / escalar_replicas for vector_valores in matriz_valores]
        escalar_mejor = min(range(len(vector_nombres)), key=vector_medias.__getitem__)
        for escalar_indice, diccionario in enumerate(vector_politicas):
            diccionario[txt_metrica] = _estadisticos(matriz_valores[escalar_indice])
            # Comparación pareada con el escenario de menor promedio: todos corren las mismas secuencias en cada réplica,
            # así que se resume la diferencia réplica a réplica. Si su intervalo incluye el cero, la diferencia no es significativa
            diccionario[txt_metrica]["vs_mejor"] = None if escalar_indice == escalar_mejor else _estadisticos(
                [escalar_valor - escalar_base for escalar_valor, escalar_base in zip(matriz_valores[escalar_indice], matriz_valores[escalar_mejor])])

    return {"replicas": escalar_replicas, "politicas": vector_politicas}
