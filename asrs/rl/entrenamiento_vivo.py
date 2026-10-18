"""Proceso que entrena un agente y va emitiendo su estado como líneas JSON, para la vista Entrenamiento del visualizador.

Se ejecuta como un proceso aparte (python -m asrs.rl.entrenamiento_vivo config.json --algoritmo ppo ...) y el servidor lee su salida.
Cada línea es un objeto JSON con un campo "tipo": inicio, referencia, metrica, snapshot, evaluacion, fin o error.
"""
import argparse
import json
import sys
import time
import traceback

import numpy as np

from asrs.config import config_desde_json
from asrs.rl.entrenamiento import CLASICAS, DESPLAZAMIENTO_PRUEBA, SEMILLA_BASE

INTERVALO_METRICAS = 1.5       # segundos entre mediciones
INTERVALO_SNAPSHOT = 4.0       # segundos entre fotos de la red


def emitir(dic_objeto):
    print(json.dumps(dic_objeto, separators=(",", ":")), flush=True)


def _redondear(vector, escalar_decimales=3):
    return np.round(np.asarray(vector, dtype=float), escalar_decimales).tolist()


def pares_capas(modelo, txt_algoritmo):
    # Capas lineales de la red de decisión con su función de activación (None en la de salida)
    import torch.nn as nn
    if txt_algoritmo == "dqn":
        vector_modulos = list(modelo.q_net.q_net)
    else:
        vector_modulos = list(modelo.policy.mlp_extractor.policy_net) + [modelo.policy.action_net]
    vector_pares, escalar_i = [], 0
    while escalar_i < len(vector_modulos):
        if isinstance(vector_modulos[escalar_i], nn.Linear):
            siguiente = vector_modulos[escalar_i + 1] if escalar_i + 1 < len(vector_modulos) else None
            if siguiente is not None and not isinstance(siguiente, nn.Linear):
                vector_pares.append((vector_modulos[escalar_i], siguiente)); escalar_i += 2
            else:
                vector_pares.append((vector_modulos[escalar_i], None)); escalar_i += 1
        else:
            escalar_i += 1
    return vector_pares


def foto_de_la_red(modelo, txt_algoritmo, vector_observacion, vector_valido, escalar_canales, escalar_pasos):
    # Estado de la red ante una observación fija: lo que ve, cómo se activan las neuronas, qué decide y cómo son sus pesos
    import torch
    import torch.nn.functional as F
    vector_pares = pares_capas(modelo, txt_algoritmo)
    with torch.no_grad():
        x = torch.as_tensor(vector_observacion, dtype=torch.float32).unsqueeze(0).to(modelo.device)
        vector_activaciones = []
        for lineal, activacion in vector_pares:
            x = lineal(x)
            if activacion is not None:
                x = activacion(x)
                vector_activaciones.append(x[0].cpu().numpy())
        vector_salida = x[0].cpu().numpy()

        # Salida: probabilidades (PPO, con la máscara de canales sin espacio) o valores Q (DQN)
        if txt_algoritmo == "ppo":
            vector_logits = np.where(vector_valido, vector_salida, -np.inf)
            vector_probabilidad = np.exp(vector_logits - vector_logits.max()); vector_probabilidad /= vector_probabilidad.sum()
            vector_valores = [None if not v else float(round(p, 5)) for p, v in zip(vector_probabilidad, vector_valido)]
            escalar_elegida = int(np.argmax(vector_probabilidad))
        else:
            vector_valores = [None if not v else float(round(q, 3)) for q, v in zip(vector_salida, vector_valido)]
            escalar_elegida = int(np.argmax(np.where(vector_valido, vector_salida, -np.inf)))

        vector_capas_pesos = []
        for escalar_indice, (lineal, _) in enumerate(vector_pares):
            matriz = lineal.weight.detach().cpu()
            vector_histograma, vector_bordes = np.histogram(matriz.numpy().ravel(), bins=24)
            dic_capa = {"salida": int(matriz.shape[0]), "entrada": int(matriz.shape[1]), "media_abs": round(float(matriz.abs().mean()), 5),
                        "desv": round(float(matriz.std()), 5), "hist": vector_histograma.tolist(),
                        "min": round(float(vector_bordes[0]), 5), "max": round(float(vector_bordes[-1]), 5)}
            if escalar_indice == 0:
                # Atención de cada neurona oculta a cada grupo de entradas: cuatro mapas por canal y los escalares
                escalar_grupos = (matriz.shape[1] - 2) // escalar_canales
                matriz_abs = matriz.abs().numpy()
                vector_columnas = [matriz_abs[:, g * escalar_canales:(g + 1) * escalar_canales].mean(axis=1) for g in range(escalar_grupos)]
                vector_columnas.append(matriz_abs[:, escalar_grupos * escalar_canales:].mean(axis=1))
                dic_capa["atencion"] = _redondear(np.stack(vector_columnas, axis=1).ravel(), 4)        # (neuronas, grupos + 1)
                dic_capa["grupos"] = escalar_grupos + 1
            elif escalar_indice < len(vector_pares) - 1:
                dic_capa["pool"] = _redondear(F.adaptive_avg_pool2d(matriz[None, None], (32, 32))[0, 0].numpy().ravel(), 4)    # mapa 32x32 de los pesos
            else:
                dic_capa["por_canal"] = _redondear(matriz.mean(dim=1).numpy(), 4)                     # peso medio hacia cada canal de salida
            vector_capas_pesos.append(dic_capa)

    escalar_c = escalar_canales
    return {"tipo": "snapshot", "pasos": int(escalar_pasos),
            "entrada": [np.round(vector_observacion[g * escalar_c:(g + 1) * escalar_c] * 255).astype(int).tolist() for g in range(4)],
            "escalares": _redondear(vector_observacion[4 * escalar_c:], 3),
            "activaciones": [_redondear(v, 3) for v in vector_activaciones],
            "salida": {"tipo": "prob" if txt_algoritmo == "ppo" else "q", "valores": vector_valores, "elegida": escalar_elegida},
            "pesos": vector_capas_pesos}


def main():
    analizador = argparse.ArgumentParser()
    analizador.add_argument("config")
    analizador.add_argument("--algoritmo", choices=("dqn", "ppo"), required=True)
    analizador.add_argument("--pasos", type=int, default=20_000)
    analizador.add_argument("--entornos", type=int, default=4)
    analizador.add_argument("--estados", type=int, default=16)
    analizador.add_argument("--pruebas", type=int, default=4)
    analizador.add_argument("--semilla", type=int, default=0)
    analizador.add_argument("--dispositivo", default="cpu")
    argumentos = analizador.parse_args()

    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    from asrs.rl import entrenamiento
    from asrs.rl.entorno import EntornoAsignacion
    from asrs.rl.evaluacion import evaluar_agente, evaluar_politica_clasica

    with open(argumentos.config, encoding="utf-8") as archivo:
        config = config_desde_json(json.load(archivo))
    vector_pruebas = list(range(DESPLAZAMIENTO_PRUEBA, DESPLAZAMIENTO_PRUEBA + argumentos.pruebas))

    # Observación fija con la que se muestra la red: la primera decisión de un estado de entrenamiento
    entorno_sonda = EntornoAsignacion(config, SEMILLA_BASE, argumentos.estados)
    vector_observacion, _ = entorno_sonda.reset(options={"estado": 0})
    vector_valido = entorno_sonda.action_masks()
    escalar_canales = entorno_sonda.escalar_canales
    emitir({"tipo": "inicio", "algoritmo": argumentos.algoritmo, "filas": entorno_sonda.escalar_filas, "columnas": entorno_sonda.escalar_columnas,
            "capas": entorno_sonda.escalar_capas, "entornos": argumentos.entornos, "pasos": argumentos.pasos, "entrada": int(vector_observacion.shape[0])})

    # Referencias: cuánto tardan las políticas clásicas en los mismos estados de prueba
    dic_estados = {}
    dic_referencias = {}
    for txt_clave in CLASICAS:
        dic_referencias[txt_clave] = evaluar_politica_clasica(config, txt_clave, SEMILLA_BASE, vector_pruebas, dic_estados)
        emitir({"tipo": "referencia", "clave": txt_clave, "media": round(float(dic_referencias[txt_clave]["tiempos"].mean()), 1)})

    class Vivo(BaseCallback):
        def __init__(self):
            super().__init__()
            self.inicio = self.ultima_metrica = self.ultima_foto = time.perf_counter()
            self.tiempos, self.episodios, self.ultimos = [], 0, {}

        def preparar(self, modelo):
            # Se registra lo que el algoritmo anota (pérdidas, entropía...) porque con verbose=0 se descarta
            ultimos = self.ultimos

            class RegistroVivo(Logger):
                def record(self, clave, valor, exclude=None):
                    if isinstance(valor, (int, float, np.floating)):
                        ultimos[clave] = float(valor)
                    super().record(clave, valor, exclude)

            modelo.set_logger(RegistroVivo(None, []))

        def _foto(self):
            emitir(foto_de_la_red(self.model, argumentos.algoritmo, vector_observacion, vector_valido, escalar_canales, self.num_timesteps))
            self.ultima_foto = time.perf_counter()

        def _on_training_start(self):
            self._foto()

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "tiempo_total" in info:
                    self.tiempos.append(info["tiempo_total"]); self.episodios += 1
            escalar_ahora = time.perf_counter()
            if escalar_ahora - self.ultima_metrica >= INTERVALO_METRICAS:
                self.ultima_metrica = escalar_ahora
                vector_reciente = self.tiempos[-30:]
                dic_metrica = {"tipo": "metrica", "pasos": int(self.num_timesteps), "seg": round(escalar_ahora - self.inicio, 1),
                               "pps": round(self.num_timesteps / max(escalar_ahora - self.inicio, 1e-9), 1), "episodios": self.episodios,
                               "tiempo": round(float(np.mean(vector_reciente)), 1) if vector_reciente else None,
                               "perdidas": {k.replace("train/", ""): round(v, 5) for k, v in self.ultimos.items() if k.startswith("train/")}}
                if argumentos.algoritmo == "dqn":
                    dic_metrica["epsilon"] = round(float(self.model.exploration_rate), 3)
                emitir(dic_metrica)
            if escalar_ahora - self.ultima_foto >= INTERVALO_SNAPSHOT:
                self._foto()
            return True

    modelo, escalar_segundos = entrenamiento.entrenar(argumentos.algoritmo, config, argumentos.pasos, argumentos.estados, argumentos.semilla,
                                                      argumentos.entornos, argumentos.dispositivo, [Vivo()])
    emitir(foto_de_la_red(modelo, argumentos.algoritmo, vector_observacion, vector_valido, escalar_canales, argumentos.pasos))

    resultado = evaluar_agente(modelo, argumentos.algoritmo, EntornoAsignacion(config, SEMILLA_BASE, argumentos.estados), vector_pruebas)
    emitir({"tipo": "evaluacion", "clave": argumentos.algoritmo.upper(), "media": round(float(resultado["tiempos"].mean()), 1)})
    import os
    os.makedirs("modelos", exist_ok=True)
    modelo.save(os.path.join("modelos", argumentos.algoritmo + "_vivo"))
    emitir({"tipo": "fin", "segundos": round(escalar_segundos, 1), "pps": round(argumentos.pasos / escalar_segundos, 1)})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        emitir({"tipo": "error", "mensaje": traceback.format_exc()[-1500:]})
        sys.exit(1)
