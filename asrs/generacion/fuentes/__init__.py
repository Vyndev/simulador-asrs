"""Tratamiento de las fuentes de datos reales: de cada fuente se extrae solo la secuencia de demanda (los SKU recuperados, en orden de tiempo, una recuperación por unidad de cada línea).

Cada fuente tiene un adaptador (un módulo de este paquete) que dice dónde están el SKU y el tiempo y qué filas no son demanda.
El resto es común: ordenar por tiempo, descartar el tiempo, contar lo filtrado y escribir un CSV de una columna (`SKU`).
Para añadir una fuente basta escribir otro adaptador y registrarlo en `registro.py`.
"""
from asrs.generacion.fuentes.base import Adaptador, Registro, leer_cantidad, ResultadoExtraccion, extraer_secuencia, escribir_secuencia, muestrear_surtido, escalar_secuencia, escribir_codigos, codificar_skus
from asrs.generacion.fuentes.registro import ADAPTADORES, crear_adaptador

__all__ = ["Adaptador", "Registro", "leer_cantidad", "ResultadoExtraccion", "extraer_secuencia", "muestrear_surtido", "escalar_secuencia", "escribir_codigos", "codificar_skus", "escribir_secuencia", "ADAPTADORES", "crear_adaptador"]
