"""Punto de entrada: ejecuta la simulación y guarda resultados_simulacion.csv."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del proyecto, para importar asrs
import time

from asrs.simulacion.simulacion import Simulacion


if __name__ == "__main__":
    escalar_tiempo_inicio = time.time()
    Simulacion().ejecutarSimulacion()
    escalar_tiempo_fin = time.time()
    escalar_duracion = (escalar_tiempo_fin - escalar_tiempo_inicio) / 60

    print(f'La simulación fue efectuada en {escalar_duracion:.2f} minutos.')
