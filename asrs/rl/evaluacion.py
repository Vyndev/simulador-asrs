"""Evaluación de agentes y de las políticas clásicas desde los mismos estados iniciales."""
import random as rnd
import time

import numpy as np

from asrs.control.controlador_grua import ControladorGrua
from asrs.politicas.recuperacion import PoliticaRecuperacionFCFS
from asrs.politicas.registro import crear_politica
from asrs.rl.entorno import copiar_estado, generar_estado


def evaluar_politica_clasica(config, txt_clave_politica, escalar_semilla_base, vector_estados, dic_estados=None):
    # Ejecuta la lista de tareas con una política clásica desde cada estado inicial, devuelve los tiempos totales y el tiempo de cálculo
    vector_tiempos, vector_omitidas = [], []
    escalar_inicio = time.perf_counter()
    for escalar_indice in vector_estados:
        if dic_estados is not None and escalar_indice not in dic_estados:
            dic_estados[escalar_indice] = generar_estado(config, escalar_semilla_base + escalar_indice)     # se genera una vez y se reutiliza con todas las políticas
        E, G, P, S = copiar_estado(dic_estados[escalar_indice] if dic_estados is not None else generar_estado(config, escalar_semilla_base + escalar_indice))
        rnd.seed(escalar_semilla_base * 7919 + escalar_indice)
        politica = crear_politica(txt_clave_politica, E, G, P, config)
        C = ControladorGrua(E, G, P, S, politica, PoliticaRecuperacionFCFS())
        C.ejecutarSecuencia()
        vector_resultados = C.instancia_metricas.recuperarInformcion()
        vector_tiempos.append(float(vector_resultados[-1]))
        vector_omitidas.append(len(S) - int(vector_resultados[-2]))          # tareas que no se pudieron ejecutar (sin espacio o sin existencias)
    return {"tiempos": np.array(vector_tiempos), "omitidas": np.array(vector_omitidas), "segundos": time.perf_counter() - escalar_inicio}


def _accion_dqn_enmascarada(modelo, vector_observacion, vector_valido):
    # Mejor acción según los valores Q, considerando solo los canales con celda disponible
    import torch
    with torch.no_grad():
        tensor = torch.as_tensor(vector_observacion, dtype=torch.float32).unsqueeze(0).to(modelo.device)
        vector_q = modelo.q_net(tensor).cpu().numpy()[0]
    return int(np.argmax(np.where(vector_valido, vector_q, -np.inf)))


def evaluar_agente(modelo, txt_algoritmo, entorno, vector_estados):
    # Ejecuta el agente entrenado sin exploración desde cada estado inicial
    # PPO usa su máscara de acciones y DQN toma el mejor valor Q entre los canales disponibles
    vector_tiempos, vector_omitidas, escalar_decisiones, escalar_invalidas = [], [], 0, 0
    escalar_inicio = time.perf_counter()
    for escalar_indice in vector_estados:
        vector_observacion, _ = entorno.reset(options={"estado": escalar_indice})
        while not entorno.terminado:
            if txt_algoritmo == "dqn":
                escalar_accion = _accion_dqn_enmascarada(modelo, vector_observacion, entorno.action_masks())
            else:
                escalar_accion, _ = modelo.predict(vector_observacion, action_masks=entorno.action_masks(), deterministic=True)
            vector_observacion, _, _, _, _ = entorno.step(int(escalar_accion))
        vector_tiempos.append(entorno.tiempo_total())
        vector_omitidas.append(len(entorno.S) - int(entorno.C.instancia_metricas.recuperarInformcion()[-2]))
        escalar_decisiones += entorno.escalar_decisiones
        escalar_invalidas += entorno.escalar_invalidas
    return {"tiempos": np.array(vector_tiempos), "omitidas": np.array(vector_omitidas), "decisiones": escalar_decisiones, "invalidas": escalar_invalidas,
            "segundos": time.perf_counter() - escalar_inicio}
