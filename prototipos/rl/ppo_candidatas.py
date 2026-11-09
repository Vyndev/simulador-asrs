"""Opción B del prototipo: PPO de un solo archivo al estilo CleanRL (Huang et al., 2022, ppo.py), adaptado a la acción del capítulo 9.

Cambios respecto de ppo.py de CleanRL, y solo estos:
1. Red por candidata (ec. 9.8): y_c = f(s_t, z_c; w) con los mismos pesos para todas las celdas; la política es una categórica sobre las
   puntuaciones y las celdas que no son candidatas reciben -inf (máscara). El crítico usa s_t y el promedio de las capas ocultas de las
   candidatas, de modo que su entrada tiene tamaño fijo.
2. Un solo entorno (sin entornos vectorizados) y la máscara se guarda con cada transición para recalcular la log-probabilidad al optimizar.
Los hiperparámetros son los de Stable-Baselines3 por defecto, para que la comparación con la opción A dependa solo de la red y la implementación.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from entorno import DIM_CANDIDATA, DIM_ESTADO


class RedCandidatas(nn.Module):
    def __init__(self, escalar_celdas, escalar_oculta=64):
        super().__init__()
        self.escalar_celdas = escalar_celdas
        self.codificador = nn.Sequential(nn.Linear(DIM_ESTADO + DIM_CANDIDATA, escalar_oculta), nn.Tanh(),
                                         nn.Linear(escalar_oculta, escalar_oculta), nn.Tanh())
        self.puntuacion = nn.Linear(escalar_oculta, 1)
        self.critico = nn.Sequential(nn.Linear(DIM_ESTADO + escalar_oculta, escalar_oculta), nn.Tanh(),
                                     nn.Linear(escalar_oculta, escalar_oculta), nn.Tanh(), nn.Linear(escalar_oculta, 1))

    def _partes(self, obs, mascara):
        estado = obs[:, :DIM_ESTADO]
        celdas = obs[:, DIM_ESTADO:].reshape(-1, self.escalar_celdas, DIM_CANDIDATA)
        entrada = torch.cat([estado.unsqueeze(1).expand(-1, self.escalar_celdas, -1), celdas], dim=2)
        oculta = self.codificador(entrada)
        logits = self.puntuacion(oculta).squeeze(-1).masked_fill(~mascara, -1e9)
        peso = mascara.float().unsqueeze(-1)
        resumen = (oculta * peso).sum(1) / peso.sum(1).clamp(min=1.0)
        valor = self.critico(torch.cat([estado, resumen], dim=1)).squeeze(-1)
        return logits, valor

    def actuar(self, obs, mascara, accion=None):
        logits, valor = self._partes(obs, mascara)
        distribucion = torch.distributions.Categorical(logits=logits)
        if accion is None:
            accion = distribucion.sample()
        return accion, distribucion.log_prob(accion), distribucion.entropy(), valor

    def mejor(self, obs, mascara):
        return self._partes(obs, mascara)[0].argmax(1)


def entrenar(env, escalar_pasos, semilla=0, n_steps=2048, batch_size=64, n_epochs=10, lr=3e-4, gamma=0.99, gae_lambda=0.95,
             clip=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, al_reportar=None):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    N = env.action_space.n
    red = RedCandidatas(N)
    optimizador = torch.optim.Adam(red.parameters(), lr=lr, eps=1e-5)
    dim_obs = env.observation_space.shape[0]
    obs_b = torch.zeros((n_steps, dim_obs)); masc_b = torch.zeros((n_steps, N), dtype=torch.bool)
    acc_b = torch.zeros(n_steps, dtype=torch.long); logp_b = torch.zeros(n_steps); rec_b = torch.zeros(n_steps)
    fin_b = torch.zeros(n_steps); val_b = torch.zeros(n_steps)
    obs, _ = env.reset(); obs = torch.tensor(obs); fin = 0.0
    paso, t0, episodios = 0, time.time(), []
    while paso < escalar_pasos:
        for i in range(n_steps):
            mascara = torch.tensor(env.action_masks())
            obs_b[i], masc_b[i], fin_b[i] = obs, mascara, fin
            with torch.no_grad():
                accion, logp, _, valor = red.actuar(obs.unsqueeze(0), mascara.unsqueeze(0))
            acc_b[i], logp_b[i], val_b[i] = accion, logp, valor
            obs_sig, recompensa, terminado, truncado, info = env.step(int(accion))
            rec_b[i] = recompensa
            paso += 1
            if terminado or truncado:
                episodios.append(info)
                obs_sig, _ = env.reset()
                fin = 1.0
            else:
                fin = 0.0
            obs = torch.tensor(obs_sig)
        # GAE, como en CleanRL
        with torch.no_grad():
            valor_sig = red.actuar(obs.unsqueeze(0), torch.tensor(env.action_masks()).unsqueeze(0))[3][0]
            ventaja = torch.zeros(n_steps); acumulada = 0.0
            for i in reversed(range(n_steps)):
                no_fin = 1.0 - (fin if i == n_steps - 1 else fin_b[i + 1])
                siguiente = valor_sig if i == n_steps - 1 else val_b[i + 1]
                delta = rec_b[i] + gamma * siguiente * no_fin - val_b[i]
                acumulada = delta + gamma * gae_lambda * no_fin * acumulada
                ventaja[i] = acumulada
            retorno = ventaja + val_b
        indices = np.arange(n_steps)
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for inicio in range(0, n_steps, batch_size):
                mb = indices[inicio:inicio + batch_size]
                _, logp_nueva, entropia, valor_nuevo = red.actuar(obs_b[mb], masc_b[mb], acc_b[mb])
                razon = (logp_nueva - logp_b[mb]).exp()
                v = ventaja[mb]; v = (v - v.mean()) / (v.std() + 1e-8)
                perdida_pol = torch.max(-v * razon, -v * razon.clamp(1 - clip, 1 + clip)).mean()
                perdida_val = 0.5 * ((valor_nuevo - retorno[mb]) ** 2).mean()
                perdida = perdida_pol + vf_coef * perdida_val - ent_coef * entropia.mean()
                optimizador.zero_grad(); perdida.backward()
                nn.utils.clip_grad_norm_(red.parameters(), max_grad_norm); optimizador.step()
        if al_reportar is not None:
            al_reportar(paso, time.time() - t0, episodios)
    return red, time.time() - t0, episodios


def decisor(red):
    def decidir(obs, mascara):
        with torch.no_grad():
            return int(red.mejor(torch.tensor(obs).unsqueeze(0), torch.tensor(mascara).unsqueeze(0))[0])
    return decidir
