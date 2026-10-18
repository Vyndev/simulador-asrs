"""Parámetros del modelo de simulación."""
import numpy as np


def cargar_configuracion():
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
        "vector_capacidad_clase": np.array([0.2, 0.4, 0.4]), # Proporción de celdas que ocupa cada clase en las políticas con zonas fijas
        "vector_demanda_clase": np.array([0.2, 0.4, 0.4]), # Proporción de la demanda (llegadas y solicitudes) de cada clase, ordena las clases en full-turnover
        "vector_prob_acum_AR": np.array([0.5, 1]), # Vector de probabilidad acumulada, para el almacenamiento y recuperación
        "escalar_largo_secuencia": 1,  # Valor inicial, se ajustará en la inicialización
        "escalar_cantidad_replicas": 100, # la simulación se repetira según la cantidad de replicas
        "escalar_semilla": 2024, # Semilla base, la réplica i usa semilla + i, esto hace reproducibles los resultados

        # Políticas de asignación de almacenamiento a comparar, también gobiernan la reubicación
        # Disponibles: 'aleatoria', 'basada_en_clases', 'full_turnover', 'mas_cercana_es', 'mas_cercana_grua'
        "vector_politicas_asignacion": ('aleatoria', 'basada_en_clases', 'full_turnover', 'mas_cercana_es', 'mas_cercana_grua'),
    }
    # Se calcula el largo de secuencia
    config["escalar_largo_secuencia"] = round(np.prod(config["vector_configuracion_estanteria"]) * 0.1)

    return config
