"""Resumen estadístico de varias réplicas: promedio, dispersión y cuántas veces gana cada política."""
import math

# Posición de cada métrica en una fila de resultados (la fila empieza con el nombre de la política)
COLUMNAS_METRICAS = {
    "tiempo_total": -1,
    "tiempo_almacenamiento": 2,
    "tiempo_recuperacion": 5,
    "tiempo_reubicacion": 8,
    "reubicaciones": 7,
    "operaciones": -2,
}


def _estadisticos(vector_valores):
    escalar_n = len(vector_valores)
    escalar_media = sum(vector_valores) / escalar_n
    escalar_desviacion = math.sqrt(sum((v - escalar_media) ** 2 for v in vector_valores) / (escalar_n - 1)) if escalar_n > 1 else 0.0
    return {
        "media": escalar_media,
        "desviacion": escalar_desviacion,
        "minimo": min(vector_valores),
        "maximo": max(vector_valores),
        # Intervalo de confianza del 95 % de la media, aproximación normal
        "ic95": 1.96 * escalar_desviacion / math.sqrt(escalar_n) if escalar_n > 1 else 0.0,
    }


def resumir(resultados_por_replica):
    # resultados_por_replica: lista con una entrada por réplica; cada entrada es la lista de filas de esa réplica, una por política
    # Devuelve un diccionario con las estadísticas por política y el conteo de victorias (comparación pareada por réplica)
    escalar_replicas = len(resultados_por_replica)
    if escalar_replicas == 0:
        return {"replicas": 0, "politicas": []}

    vector_nombres = [fila[0] for fila in resultados_por_replica[0]]
    vector_victorias = [0] * len(vector_nombres)
    for vector_filas in resultados_por_replica:
        vector_tiempos = [float(fila[-1]) for fila in vector_filas]
        escalar_mejor = min(vector_tiempos)
        for escalar_indice, escalar_tiempo in enumerate(vector_tiempos):
            if escalar_tiempo <= escalar_mejor + 1e-9:
                vector_victorias[escalar_indice] += 1

    vector_politicas = []
    for escalar_indice, txt_nombre in enumerate(vector_nombres):
        diccionario = {"nombre": txt_nombre, "victorias": vector_victorias[escalar_indice]}
        for txt_metrica, escalar_columna in COLUMNAS_METRICAS.items():
            diccionario[txt_metrica] = _estadisticos([float(vector_filas[escalar_indice][escalar_columna]) for vector_filas in resultados_por_replica])
        vector_politicas.append(diccionario)

    return {"replicas": escalar_replicas, "politicas": vector_politicas}
