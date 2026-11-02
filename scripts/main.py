"""Punto de entrada: ejecuta la simulación y guarda salidas/resultados_simulacion.csv."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del proyecto, para importar asrs
import argparse
import time

from asrs.config import cargar_configuracion, cargar_configuracion_archivo
from asrs.simulacion.simulacion import Simulacion


if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--config", default="", help="experimento en JSON (ver experimentos/); sin él se usa asrs/config.py")
    argumentos = analizador.parse_args()
    config = cargar_configuracion_archivo(argumentos.config) if argumentos.config else cargar_configuracion()

    escalar_tiempo_inicio = time.time()
    Simulacion(config).ejecutarSimulacion()
    escalar_tiempo_fin = time.time()
    escalar_duracion = (escalar_tiempo_fin - escalar_tiempo_inicio) / 60

    print(f'La simulación fue efectuada en {escalar_duracion:.2f} minutos.')
