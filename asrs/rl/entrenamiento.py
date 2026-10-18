"""Entrenamiento de DQN y PPO para la asignación de almacenamiento, compartido por el script y la vista en vivo."""
import time
from functools import partial

CLASICAS = ('aleatoria', 'basada_en_clases', 'full_turnover', 'mas_cercana_es', 'mas_cercana_grua')
SEMILLA_BASE = 1000                    # los estados de entrenamiento son 0..N-1 y los de prueba empiezan en DESPLAZAMIENTO_PRUEBA
DESPLAZAMIENTO_PRUEBA = 500


def crear_entornos(config, escalar_estados, escalar_semilla, escalar_entornos):
    # Un entorno por proceso: la simulación es lo más lento y así usa varios núcleos
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from asrs.rl.entorno import crear_entorno_monitoreado
    fabricas = [partial(crear_entorno_monitoreado, config, SEMILLA_BASE, escalar_estados, escalar_semilla * 1000 + i) for i in range(escalar_entornos)]
    return SubprocVecEnv(fabricas) if escalar_entornos > 1 else DummyVecEnv(fabricas)


def crear_modelo(txt_algoritmo, entorno, escalar_entornos, escalar_semilla, txt_dispositivo="cpu"):
    from stable_baselines3 import DQN
    from sb3_contrib import MaskablePPO
    if txt_algoritmo == "dqn":
        # Se mantiene la proporción de un paso de aprendizaje por cada 4 transiciones, sin importar cuántos entornos haya
        return DQN("MlpPolicy", entorno, learning_rate=5e-4, buffer_size=50_000, learning_starts=1_000, batch_size=128,
                   train_freq=1 if escalar_entornos > 1 else 4, gradient_steps=max(1, escalar_entornos // 4),
                   target_update_interval=max(1, 500 // escalar_entornos), exploration_fraction=0.4, exploration_final_eps=0.05,
                   gamma=0.99, policy_kwargs={"net_arch": [256, 256]}, seed=escalar_semilla, verbose=0, device=txt_dispositivo)
    # Cada actualización usa unas 2048 transiciones repartidas entre los entornos
    return MaskablePPO("MlpPolicy", entorno, learning_rate=3e-4, n_steps=max(64, 2048 // escalar_entornos), batch_size=256, n_epochs=5,
                       gamma=0.99, ent_coef=0.01, policy_kwargs={"net_arch": [256, 256]}, seed=escalar_semilla, verbose=0, device=txt_dispositivo)


def entrenar(txt_algoritmo, config, escalar_pasos, escalar_estados, escalar_semilla, escalar_entornos=1, txt_dispositivo="cpu", vector_callbacks=None):
    # Entrena y devuelve el modelo y los segundos que tardó
    entorno = crear_entornos(config, escalar_estados, escalar_semilla, escalar_entornos)
    try:
        modelo = crear_modelo(txt_algoritmo, entorno, escalar_entornos, escalar_semilla, txt_dispositivo)
        if vector_callbacks:
            for callback in vector_callbacks:
                if hasattr(callback, "preparar"):
                    callback.preparar(modelo)
        escalar_inicio = time.perf_counter()
        modelo.learn(total_timesteps=escalar_pasos, callback=vector_callbacks or None)
        return modelo, time.perf_counter() - escalar_inicio
    finally:
        entorno.close()
