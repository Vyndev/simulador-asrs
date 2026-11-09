"""Compara dos formas de entrenar el agente del capítulo 9 en el mismo entorno (prototipo, estantería 6 x 6 x 2):

  A  Stable-Baselines3: MaskablePPO de sb3-contrib con la política MLP por defecto sobre la observación aplanada.
  B  PPO de un solo archivo al estilo CleanRL con una red que puntúa cada candidata con los mismos pesos (ppo_candidatas.py).

Mismos hiperparámetros (los de Stable-Baselines3 por defecto), mismas semillas de entrenamiento y de evaluación, que son disjuntas.
Uso: python comparar.py --opcion A|B|heuristicas --pasos 200000 --semilla 0 --salida resultados_A0.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from entorno import EntornoAsignacion, config_prototipo, evaluar_agente, evaluar_heuristica

SEMILLAS_ENTRENAMIENTO = list(range(1, 5001))
SEMILLAS_EVALUACION = list(range(100_001, 100_011))
HEURISTICAS = ("aleatoria", "basada_en_clases", "full_turnover", "mas_cercana_es", "mas_cercana_grua")


def evaluar(config, decidir):
    vector = [evaluar_agente(config, decidir, s) for s in SEMILLAS_EVALUACION]
    return {"tiempo": [t for t, _ in vector], "perdidas": [p for _, p in vector]}


def resumen_episodios(episodios, escalar_ultimos=20):
    ultimos = episodios[-escalar_ultimos:]
    return {"episodios": len(episodios), "tiempo_medio_ultimos": float(np.mean([e["tiempo_total"] for e in ultimos])) if ultimos else None,
            "perdidas_medias_ultimos": float(np.mean([e["perdidas"] for e in ultimos])) if ultimos else None}


def main():
    analizador = argparse.ArgumentParser()
    analizador.add_argument("--opcion", choices=["A", "B", "heuristicas"], required=True)
    analizador.add_argument("--pasos", type=int, default=200_000)
    analizador.add_argument("--hilos", type=int, default=1)
    analizador.add_argument("--semilla", type=int, default=0, help="semilla del entrenamiento (red y muestreo de acciones)")
    analizador.add_argument("--salida", required=True)
    argumentos = analizador.parse_args()
    torch.set_num_threads(argumentos.hilos)
    config = config_prototipo()
    curva = []

    def reportar(paso, segundos, episodios):
        dic = {"paso": paso, "segundos": round(segundos, 1), **resumen_episodios(episodios)}
        curva.append(dic)
        print(json.dumps(dic), flush=True)

    if argumentos.opcion == "heuristicas":
        resultado = {h: dict(zip(("tiempo", "perdidas"), map(list, zip(*[evaluar_heuristica(config, h, s) for s in SEMILLAS_EVALUACION]))))
                     for h in HEURISTICAS}
    elif argumentos.opcion == "A":
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback

        env = EntornoAsignacion(config, SEMILLAS_ENTRENAMIENTO)

        class Curva(BaseCallback):
            def __init__(self):
                super().__init__(); self.episodios = []; self.t0 = time.time()

            def _on_step(self):
                for info in self.locals["infos"]:
                    if "tiempo_total" in info:
                        self.episodios.append(info)
                return True

            def _on_rollout_end(self):
                reportar(self.num_timesteps, time.time() - self.t0, self.episodios)

        callback = Curva()
        t0 = time.time()
        modelo = MaskablePPO("MlpPolicy", env, seed=argumentos.semilla, verbose=0, device="cpu")
        modelo.learn(total_timesteps=argumentos.pasos, callback=callback)
        segundos = time.time() - t0
        env.close()

        def decidir(obs, mascara):
            return int(modelo.predict(obs, action_masks=mascara, deterministic=True)[0])

        resultado = {"segundos_entrenamiento": segundos, "parametros": sum(p.numel() for p in modelo.policy.parameters()),
                     "evaluacion": evaluar(config, decidir), "curva": curva}
    else:
        from ppo_candidatas import decisor, entrenar
        env = EntornoAsignacion(config, SEMILLAS_ENTRENAMIENTO)
        red, segundos, _ = entrenar(env, argumentos.pasos, semilla=argumentos.semilla, al_reportar=reportar)
        env.close()
        resultado = {"segundos_entrenamiento": segundos, "parametros": sum(p.numel() for p in red.parameters()),
                     "evaluacion": evaluar(config, decisor(red)), "curva": curva}
    with open(argumentos.salida, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo)
    print("listo", argumentos.salida)


if __name__ == "__main__":
    main()
