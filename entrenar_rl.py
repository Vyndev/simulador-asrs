"""Entrena DQN y PPO para la asignación de almacenamiento y los compara con las políticas clásicas.

Uso:
  python entrenar_rl.py --algoritmo ppo --pasos 20000
  python entrenar_rl.py --algoritmo ambos --pasos 50000 --pruebas 20

Mide cuánto tarda el entrenamiento (pasos por segundo) y el desempeño en estados de prueba que el agente no vio.
Requiere: pip install -r requirements-rl.txt
"""
import argparse
import os
import time

import numpy as np

from asrs.config import cargar_configuracion

from asrs.rl import entrenamiento
from asrs.rl.entrenamiento import CLASICAS, DESPLAZAMIENTO_PRUEBA, SEMILLA_BASE


def crear_config(escalar_tareas):
    config = cargar_configuracion()
    if escalar_tareas:
        config["escalar_largo_secuencia"] = escalar_tareas
    return config


def entrenar(txt_algoritmo, config, escalar_pasos, escalar_estados, escalar_semilla, escalar_entornos=1, txt_dispositivo="cpu"):
    from stable_baselines3.common.callbacks import BaseCallback

    class ProgresoImpreso(BaseCallback):
        def __init__(self, cada):
            super().__init__()
            self.cada, self.inicio, self.ultimo = cada, time.perf_counter(), 0

        def _on_step(self):
            if self.num_timesteps - self.ultimo >= self.cada:
                self.ultimo = self.num_timesteps
                transcurrido = time.perf_counter() - self.inicio
                print(f"  {self.num_timesteps:>8} pasos | {transcurrido:6.0f} s | {self.num_timesteps / transcurrido:5.0f} pasos/s", flush=True)
            return True

    return entrenamiento.entrenar(txt_algoritmo, config, escalar_pasos, escalar_estados, escalar_semilla, escalar_entornos, txt_dispositivo,
                                  [ProgresoImpreso(max(1000, escalar_pasos // 10))])


def main():
    analizador = argparse.ArgumentParser(description="Entrena DQN y PPO y los compara con las políticas clásicas.")
    analizador.add_argument("--algoritmo", choices=("dqn", "ppo", "ambos"), default="ambos")
    analizador.add_argument("--pasos", type=int, default=20_000, help="decisiones de entrenamiento por algoritmo")
    analizador.add_argument("--estados", type=int, default=16, help="estados iniciales distintos usados para entrenar")
    analizador.add_argument("--pruebas", type=int, default=10, help="estados nuevos para evaluar")
    analizador.add_argument("--tareas", type=int, default=0, help="tareas por episodio (0 usa el 10 %% de las celdas)")
    analizador.add_argument("--semilla", type=int, default=0)
    analizador.add_argument("--entornos", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="entornos en paralelo (uno por núcleo)")
    analizador.add_argument("--dispositivo", choices=("cpu", "cuda", "auto"), default="cpu", help="dónde se entrenan las redes")
    argumentos = analizador.parse_args()

    from asrs.rl.entorno import EntornoAsignacion
    from asrs.rl.evaluacion import evaluar_agente, evaluar_politica_clasica

    config = crear_config(argumentos.tareas)
    vector_pruebas = list(range(DESPLAZAMIENTO_PRUEBA, DESPLAZAMIENTO_PRUEBA + argumentos.pruebas))
    print(f"Estantería {[int(x) for x in config['vector_configuracion_estanteria']]}, {config['escalar_largo_secuencia']} tareas por episodio, "
          f"{argumentos.estados} estados de entrenamiento, {argumentos.pruebas} de prueba.\n")

    dic_resultados = {}
    for txt_clave in CLASICAS:
        dic_resultados[txt_clave] = evaluar_politica_clasica(config, txt_clave, SEMILLA_BASE, vector_pruebas)

    for txt_algoritmo in (("dqn", "ppo") if argumentos.algoritmo == "ambos" else (argumentos.algoritmo,)):
        print(f"Entrenando {txt_algoritmo.upper()} ({argumentos.pasos} pasos, {argumentos.entornos} entornos en paralelo, {argumentos.dispositivo})...", flush=True)
        modelo, escalar_segundos = entrenar(txt_algoritmo, config, argumentos.pasos, argumentos.estados, argumentos.semilla, argumentos.entornos, argumentos.dispositivo)
        os.makedirs("modelos", exist_ok=True)
        modelo.save(os.path.join("modelos", txt_algoritmo))
        print(f"  entrenamiento: {escalar_segundos:.0f} s ({argumentos.pasos / escalar_segundos:.0f} pasos/s)")
        entorno = EntornoAsignacion(config, SEMILLA_BASE, argumentos.estados)
        resultado = evaluar_agente(modelo, txt_algoritmo, entorno, vector_pruebas)
        resultado["entrenamiento_s"] = escalar_segundos
        dic_resultados[txt_algoritmo.upper()] = resultado
        print(f"  evaluación: {resultado['segundos']:.1f} s, {1000 * resultado['segundos'] / max(1, resultado['decisiones']):.1f} ms por decisión\n")

    # Tabla comparativa: tiempo total de la lista (menor es mejor) con diferencias pareadas contra la mejor política clásica
    from asrs.rl import comparacion
    print(comparacion.tabla(dic_resultados))

if __name__ == "__main__":
    main()
