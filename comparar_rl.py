"""Compara los agentes ya entrenados (carpeta modelos/) con las políticas clásicas en estados de prueba nuevos.

Uso:
  python comparar_rl.py --pruebas 30
  python comparar_rl.py --pruebas 50 --tareas 200 --csv resultados_rl.csv

No entrena nada: usa modelos/dqn.zip y modelos/ppo.zip (o las versiones *_vivo.zip de la vista Entrenamiento) que ya existan.
"""
import argparse
import os

from asrs.config import cargar_configuracion
from asrs.rl import comparacion
from asrs.rl.entrenamiento import CLASICAS, DESPLAZAMIENTO_PRUEBA, SEMILLA_BASE


def main():
    analizador = argparse.ArgumentParser(description="Compara los agentes entrenados con las políticas clásicas.")
    analizador.add_argument("--pruebas", type=int, default=30, help="estados de prueba (cuantos más, más fiable)")
    analizador.add_argument("--tareas", type=int, default=0, help="tareas por episodio (0 usa el 10 %% de las celdas); debe coincidir con las del entrenamiento")
    analizador.add_argument("--estados", type=int, default=16, help="estados de entrenamiento con los que se entrenó el agente")
    analizador.add_argument("--csv", default="", help="guarda la tabla en un CSV")
    argumentos = analizador.parse_args()

    from stable_baselines3 import DQN
    from sb3_contrib import MaskablePPO
    from asrs.rl.entorno import EntornoAsignacion
    from asrs.rl.evaluacion import evaluar_agente, evaluar_politica_clasica

    config = cargar_configuracion()
    if argumentos.tareas:
        config["escalar_largo_secuencia"] = argumentos.tareas
    vector_pruebas = list(range(DESPLAZAMIENTO_PRUEBA, DESPLAZAMIENTO_PRUEBA + argumentos.pruebas))
    dic_estados, dic_resultados = {}, {}
    print(f"{argumentos.pruebas} estados de prueba, {config['escalar_largo_secuencia']} tareas por episodio.\n")
    for txt_clave in CLASICAS:
        dic_resultados[txt_clave] = evaluar_politica_clasica(config, txt_clave, SEMILLA_BASE, vector_pruebas, dic_estados)

    for txt_algoritmo, clase in (("dqn", DQN), ("ppo", MaskablePPO)):
        for txt_sufijo in ("", "_vivo"):
            txt_ruta = os.path.join("modelos", txt_algoritmo + txt_sufijo + ".zip")
            if os.path.exists(txt_ruta):
                try:
                    modelo = clase.load(txt_ruta, device="cpu")
                    dic_resultados[txt_algoritmo.upper() + txt_sufijo] = evaluar_agente(modelo, txt_algoritmo, EntornoAsignacion(config, SEMILLA_BASE, argumentos.estados), vector_pruebas)
                except Exception as error:
                    print(f"Se omite {txt_ruta}: no es compatible con el entorno actual ({type(error).__name__}). Vuelve a entrenarlo.")
    if len(dic_resultados) == len(CLASICAS):
        print("No hay modelos entrenados en la carpeta modelos/: ejecuta primero entrenar_rl.py o la vista Entrenamiento.\n")
    print(comparacion.tabla(dic_resultados))
    if argumentos.csv:
        comparacion.guardar_csv(dic_resultados, argumentos.csv)
        print(f"\nTabla guardada en {argumentos.csv}")


if __name__ == "__main__":
    main()
