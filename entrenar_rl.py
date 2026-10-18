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

CLASICAS = ('aleatoria', 'basada_en_clases', 'full_turnover', 'mas_cercana_es', 'mas_cercana_grua')
SEMILLA_BASE = 1000                    # los estados de entrenamiento son 0..N-1 y los de prueba empiezan en DESPLAZAMIENTO_PRUEBA
DESPLAZAMIENTO_PRUEBA = 500


def crear_config(escalar_tareas):
    config = cargar_configuracion()
    if escalar_tareas:
        config["escalar_largo_secuencia"] = escalar_tareas
    return config


def entrenar(txt_algoritmo, config, escalar_pasos, escalar_estados, escalar_semilla, escalar_entornos=1, txt_dispositivo="cpu"):
    from functools import partial

    from stable_baselines3 import DQN
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from asrs.rl.entorno import crear_entorno_monitoreado

    class Progreso(BaseCallback):
        # Imprime el avance cada cierto número de pasos
        def __init__(self, cada):
            super().__init__()
            self.cada, self.inicio, self.ultimo = cada, time.perf_counter(), 0

        def _on_step(self):
            if self.num_timesteps - self.ultimo >= self.cada:
                self.ultimo = self.num_timesteps
                transcurrido = time.perf_counter() - self.inicio
                print(f"  {self.num_timesteps:>8} pasos | {transcurrido:6.0f} s | {self.num_timesteps / transcurrido:5.0f} pasos/s", flush=True)
            return True

    # Cada entorno corre en su propio proceso, así la simulación (que es lo más lento) usa varios núcleos
    fabricas = [partial(crear_entorno_monitoreado, config, SEMILLA_BASE, escalar_estados, escalar_semilla * 1000 + i) for i in range(escalar_entornos)]
    entorno = SubprocVecEnv(fabricas) if escalar_entornos > 1 else DummyVecEnv(fabricas)
    try:
        if txt_algoritmo == "dqn":
            # Se mantiene la proporción de un paso de aprendizaje por cada 4 transiciones, sin importar cuántos entornos haya
            modelo = DQN("MlpPolicy", entorno, learning_rate=5e-4, buffer_size=50_000, learning_starts=1_000, batch_size=128,
                         train_freq=1 if escalar_entornos > 1 else 4, gradient_steps=max(1, escalar_entornos // 4),
                         target_update_interval=max(1, 500 // escalar_entornos), exploration_fraction=0.4, exploration_final_eps=0.05,
                         gamma=0.99, policy_kwargs={"net_arch": [256, 256]}, seed=escalar_semilla, verbose=0, device=txt_dispositivo)
        else:
            # Cada actualización usa unas 2048 transiciones repartidas entre los entornos
            modelo = MaskablePPO("MlpPolicy", entorno, learning_rate=3e-4, n_steps=max(64, 2048 // escalar_entornos), batch_size=256, n_epochs=5,
                                 gamma=0.99, ent_coef=0.01, policy_kwargs={"net_arch": [256, 256]}, seed=escalar_semilla, verbose=0, device=txt_dispositivo)
        escalar_inicio = time.perf_counter()
        modelo.learn(total_timesteps=escalar_pasos, callback=Progreso(max(1000, escalar_pasos // 10)))
        return modelo, time.perf_counter() - escalar_inicio
    finally:
        entorno.close()


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

    # Tabla comparativa: tiempo total de la lista (menor es mejor) y victorias pareadas contra la mejor política clásica
    escalar_mejor = min(CLASICAS, key=lambda c: dic_resultados[c]["tiempos"].mean())
    print(f"{'Política':<20}{'Tiempo total (s)':>18}{'± error':>10}{'vs mejor clásica':>20}{'Mejor que ella':>16}")
    for txt_clave, dic in dic_resultados.items():
        vector = dic["tiempos"]
        escalar_error = vector.std(ddof=1) / np.sqrt(len(vector)) if len(vector) > 1 else 0.0
        escalar_dif = 100 * (vector.mean() / dic_resultados[escalar_mejor]["tiempos"].mean() - 1)
        escalar_ganadas = int((vector < dic_resultados[escalar_mejor]["tiempos"]).sum())
        print(f"{txt_clave:<20}{vector.mean():>18.0f}{escalar_error:>10.0f}{escalar_dif:>19.1f}%{escalar_ganadas:>12}/{len(vector)}")
    print(f"\nMejor política clásica: {escalar_mejor}. Los agentes se evalúan sin exploración y los canales sin espacio se enmascaran.")


if __name__ == "__main__":
    main()
