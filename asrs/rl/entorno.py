"""Entorno de aprendizaje por refuerzo para la asignación de almacenamiento.

Formulación
  Estado  X_t = ((l, c, d), (i, e)) más la tarea actual y el perfil de la demanda:
    - Configuración física: para cada posición (nivel l, columna c, profundidad d) el SKU almacenado i y el orden de ingreso e (FIFO).
      Se completa con la cercanía de cada posición al E/S, que es la geometría de las coordenadas (l, c, d).
    - Tarea actual: el SKU i_t y el tipo de operación θ_t (almacenamiento o recuperación). Cuando la decisión es reubicar un ítem que obstruye,
      la tarea en curso es una recuperación y se informa además el SKU del ítem que hay que ubicar.
    - Perfil de la demanda: el balance p_t entre almacenamientos y recuperaciones y los perfiles de SKUs que entran P_t^S y que salen P_t^R,
      calculados con las últimas tareas (una ventana móvil) y suavizados con la demanda de planificación al inicio, cuando aún hay poco historial.
  Acción  una de las posiciones disponibles de almacenamiento (celdas libres y accesibles, es decir sin ítems delante).
  Recompensa  el tiempo de ciclo con signo negativo: lo que tarda la grúa desde una decisión hasta la siguiente, en minutos.
              El retorno del episodio es el tiempo total de la lista, la misma métrica con la que se comparan las políticas clásicas.

El agente decide dónde almacenar cada ítem de la lista y dónde reubicar cada ítem que obstruye una recuperación. Las recuperaciones son FCFS.
Si no hay ninguna posición disponible no hay decisión: la tarea se omite al almacenar y el ítem se deja temporalmente en el E/S al reubicar,
igual que en el simulador.

El episodio parte de una estantería ya llena (con una política de llenado neutra) y recorre la lista de tareas. El simulador corre en un
hilo que se detiene en cada decisión y espera la acción, así se reutiliza el controlador de la grúa sin modificarlo.
"""
import copy
import queue
import random as rnd
import threading
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from asrs.control.controlador_grua import ControladorGrua
from asrs.dominio.item import Item
from asrs.politicas.asignacion import PoliticaAsignacion
from asrs.politicas.recuperacion import PoliticaRecuperacionFCFS
from asrs.politicas.restriccion import MatricesRestriccion
from asrs.simulacion.simulacion import Simulacion

ESCALAR_SEGUNDOS_POR_UNIDAD = 60.0          # la recompensa se mide en minutos
ESCALAR_PENALIZACION_INVALIDA = 1.0         # recompensa negativa por elegir una posición no disponible
ESCALAR_MAPAS = 6                           # mapas por posición: ocupada, SKU, mismo SKU, antigüedad, disponible y cercanía al E/S
ESCALAR_PESO_PRIOR = 10.0                   # peso de la demanda de planificación en los perfiles, en tareas equivalentes


def generar_estado(config, escalar_semilla, txt_politica_llenado='aleatoria'):
    # Estado inicial de un episodio: estantería, grúa y E/S ya llenos con una política neutra, más la lista de tareas a ejecutar
    # La misma semilla produce el mismo estado, así todas las políticas pueden compararse desde el mismo punto de partida
    rnd.seed(escalar_semilla)
    instancia_simulacion = Simulacion(config)
    SA, S = instancia_simulacion.generarSecuencias()
    E, G, P, _, _ = instancia_simulacion.prepararEscenario(txt_politica_llenado, SA)
    G.escalar_reloj = 0.0
    return E, G, P, S


def copiar_estado(tupla_estado):
    # Copia profunda para que cada política parta del mismo estado sin modificar el original
    return copy.deepcopy(tupla_estado)


class EpisodioAbortado(Exception):
    # Se lanza dentro del hilo del simulador para terminarlo cuando el entorno se reinicia o se cierra
    pass


class PoliticaAgente(PoliticaAsignacion):
    # Política que delega cada decisión de ubicación en el agente a través del entorno
    nombre = 'Agente'

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, config, entorno):
        super().__init__(instancia_estanteria, instancia_grua, instancia_puntoES, config)
        self.entorno = entorno

    def asignar(self, escalar_sku, matriz_extra=None, booleano_restringir_zona=True, vector_posicion_referencia=None):
        return self.entorno._decidir(escalar_sku, matriz_extra)

    def matrizCriterio(self, vector_posicion_referencia=None):
        raise NotImplementedError("El agente decide directamente la posición.")


class EntornoAsignacion(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config, escalar_semilla_base=0, escalar_estados=16, txt_politica_llenado='aleatoria', escalar_ventana=30,
                 escalar_semilla_episodios=None):
        super().__init__()
        self.config = config
        self.escalar_semilla_base = escalar_semilla_base
        self.escalar_estados = escalar_estados
        self.txt_politica_llenado = txt_politica_llenado
        self.escalar_ventana = escalar_ventana
        self.dic_estados = {}                       # estados iniciales ya generados, se crean la primera vez que se usan
        # Con varios entornos en paralelo cada uno sortea los estados con su propia semilla, aunque todos comparten los mismos estados posibles
        self.generador = np.random.default_rng(escalar_semilla_base if escalar_semilla_episodios is None else escalar_semilla_episodios)

        escalar_filas, escalar_columnas, escalar_capas = (int(x) for x in config["vector_configuracion_estanteria"])
        self.escalar_filas, self.escalar_columnas, self.escalar_capas = escalar_filas, escalar_columnas, escalar_capas
        self.escalar_celdas = escalar_filas * escalar_columnas * escalar_capas
        self.escalar_skus = int(config["escalar_numero_skus"])
        self.vector_demanda_base = np.asarray(config["vector_demanda_sku"], dtype=float)

        # Rango de cada SKU por demanda de planificación (0 es el más demandado), normalizado
        vector_orden = np.argsort(-self.vector_demanda_base, kind='stable')
        self.vector_rango_sku = np.empty(self.escalar_skus)
        self.vector_rango_sku[vector_orden] = np.arange(self.escalar_skus)
        self.vector_rango_sku /= max(self.escalar_skus - 1, 1)

        # Observación: mapas por posición, tarea actual (5 números) y perfil de la demanda (balance, SKUs que entran, SKUs que salen)
        self.escalar_dim_observacion = ESCALAR_MAPAS * self.escalar_celdas + 5 + 1 + 2 * self.escalar_skus
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.escalar_dim_observacion,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.escalar_celdas)

        self.E = self.G = self.P = self.S = self.C = self.politica = self.hilo = None
        self.terminado = True
        self.matriz_candidatas = np.zeros((escalar_filas, escalar_columnas, escalar_capas), dtype=bool)
        self.matriz_ocupacion = np.zeros_like(self.matriz_candidatas)
        self.escalar_decisiones = 0
        self.escalar_invalidas = 0

    # ---- ciclo de vida ----
    def _estado(self, escalar_indice):
        if escalar_indice not in self.dic_estados:
            self.dic_estados[escalar_indice] = generar_estado(self.config, self.escalar_semilla_base + escalar_indice, self.txt_politica_llenado)
        return self.dic_estados[escalar_indice]

    def _detener_hilo(self):
        if self.hilo is not None and self.hilo.is_alive():
            self.cola_accion.put(None)
            self.hilo.join(timeout=5)
        self.hilo = None

    def close(self):
        self._detener_hilo()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._detener_hilo()
        # Con options={'estado': i} se elige un estado concreto, si no se sortea uno del conjunto
        escalar_indice = options["estado"] if options and "estado" in options else int(self.generador.integers(self.escalar_estados))
        self.E, self.G, self.P, self.S = copiar_estado(self._estado(escalar_indice))
        self.politica = PoliticaAgente(self.E, self.G, self.P, self.config, self)
        self.C = ControladorGrua(self.E, self.G, self.P, self.S, self.politica, PoliticaRecuperacionFCFS())
        self.historial = deque(maxlen=self.escalar_ventana)
        self.cola_solicitud, self.cola_accion = queue.Queue(), queue.Queue()
        self.escalar_decisiones = self.escalar_invalidas = 0
        self.escalar_penalizacion = 0.0
        self.escalar_t_anterior = self.G.escalar_reloj
        self.tupla_tarea, self.txt_contexto, self.escalar_sku_a_ubicar, self.error_hilo = (0, 1), 'almacenar', 1, None
        self.terminado = False
        self.hilo = threading.Thread(target=self._correr, daemon=True)
        self.hilo.start()
        self._esperar_decision()
        return self._observacion(), {}

    # ---- hilo del simulador ----
    def _correr(self):
        # Replica el recorrido de ControladorGrua.ejecutarSecuencia, con la decisión de ubicar en manos del agente
        try:
            for escalar_actividad, escalar_sku in self.S:
                self.tupla_tarea = (escalar_actividad, escalar_sku)
                if escalar_actividad == 0:
                    self.txt_contexto = 'almacenar'
                    vector_indice = self.politica.asignar(escalar_sku)
                    self.txt_contexto = 'reubicar'
                    if len(vector_indice):
                        self.P.matriz_contenedor_temporal_items.append(Item(escalar_sku))
                        self.C.almacenar(vector_indice)
                else:
                    self.txt_contexto = 'reubicar'
                    vector_indice = self.C.instancia_politica_recuperacion.seleccionar(self.E, escalar_sku)
                    if len(vector_indice):
                        self.C.recuperar(vector_indice)
                self.historial.append((escalar_actividad, escalar_sku))
        except EpisodioAbortado:
            return
        except Exception as error:
            self.error_hilo = error
        self.cola_solicitud.put(None)

    def _decidir(self, escalar_sku, matriz_extra):
        # Lo llama la política del agente desde el hilo: publica la decisión pendiente y espera la acción
        matriz_ocupacion = MatricesRestriccion().ocupacion(self.E)
        matriz_candidatas = ~matriz_ocupacion & MatricesRestriccion().accesibilidad(matriz_ocupacion)
        if matriz_extra is not None:
            matriz_candidatas = matriz_candidatas & matriz_extra
        if not matriz_candidatas.any():
            return ()
        self.matriz_ocupacion, self.matriz_candidatas, self.escalar_sku_a_ubicar = matriz_ocupacion, matriz_candidatas, escalar_sku
        self.cola_solicitud.put(True)
        escalar_accion = self.cola_accion.get()
        if escalar_accion is None:
            raise EpisodioAbortado()
        vector_disponible = matriz_candidatas.reshape(-1)
        if not vector_disponible[escalar_accion]:
            # Posición no disponible: se usa la disponible más cercana al E/S y se penaliza
            self.escalar_invalidas += 1
            self.escalar_penalizacion += ESCALAR_PENALIZACION_INVALIDA
            vector_tiempos = self.politica.tiempoChebyshevXY(self.P.vector_posicion).reshape(-1)
            escalar_accion = int(np.argmin(np.where(vector_disponible, vector_tiempos, np.inf)))
        return tuple(int(v) for v in np.unravel_index(escalar_accion, matriz_candidatas.shape))

    def _esperar_decision(self):
        self.terminado = self.cola_solicitud.get() is None
        if self.error_hilo is not None:
            raise self.error_hilo

    # ---- observación ----
    def _observacion(self):
        if self.terminado:
            return np.zeros(self.escalar_dim_observacion, dtype=np.float32)
        escalar_celdas = self.escalar_celdas
        vector_ocupada = self.matriz_ocupacion.reshape(-1)
        vector_items = self.E.matriz_contenedor_items.reshape(-1)
        vector_sku, vector_mismo, vector_edad = np.zeros(escalar_celdas), np.zeros(escalar_celdas), np.zeros(escalar_celdas)
        vector_indices = np.flatnonzero(vector_ocupada)
        vector_ingreso = np.empty(len(vector_indices))
        for escalar_k, escalar_i in enumerate(vector_indices):
            item = vector_items[escalar_i]
            vector_sku[escalar_i] = self.vector_rango_sku[item.escalar_sku - 1]
            vector_mismo[escalar_i] = 1.0 if item.escalar_sku == self.escalar_sku_a_ubicar else 0.0
            vector_ingreso[escalar_k] = item.escalar_etiqueta_generacion
        if len(vector_indices):
            # Orden de ingreso e relativo al inventario: 1 es el ítem más antiguo, el primero en salir con FIFO
            vector_edad[vector_indices] = (vector_ingreso.max() - vector_ingreso) / max(vector_ingreso.max() - vector_ingreso.min(), 1.0)
        # Cercanía al E/S de cada posición, normalizada (1 es la más cercana)
        vector_tiempos = self.politica.tiempoChebyshevXY(self.P.vector_posicion).reshape(-1)
        vector_cercania = 1.0 - vector_tiempos / max(float(vector_tiempos.max()), 1e-9)

        # Tarea actual: SKU, tipo de operación, si la decisión es una reubicación y el SKU del ítem que hay que ubicar
        escalar_actividad, escalar_sku = self.tupla_tarea
        vector_tarea = [self.vector_rango_sku[escalar_sku - 1], 1.0 if escalar_actividad == 0 else 0.0, 1.0 if escalar_actividad == 1 else 0.0,
                        1.0 if self.txt_contexto != 'almacenar' else 0.0, self.vector_rango_sku[self.escalar_sku_a_ubicar - 1]]

        # Perfil de la demanda con las últimas tareas, suavizado con la demanda de planificación
        vector_entran, vector_salen = np.zeros(self.escalar_skus), np.zeros(self.escalar_skus)
        escalar_entran = escalar_salen = 0
        for escalar_act, escalar_s in self.historial:
            if escalar_act == 0:
                vector_entran[escalar_s - 1] += 1; escalar_entran += 1
            else:
                vector_salen[escalar_s - 1] += 1; escalar_salen += 1
        vector_perfil_entran = (vector_entran + ESCALAR_PESO_PRIOR * self.vector_demanda_base) / (escalar_entran + ESCALAR_PESO_PRIOR)
        vector_perfil_salen = (vector_salen + ESCALAR_PESO_PRIOR * self.vector_demanda_base) / (escalar_salen + ESCALAR_PESO_PRIOR)
        escalar_balance = (escalar_entran + 5.0) / (escalar_entran + escalar_salen + 10.0)

        return np.concatenate([vector_ocupada.astype(float), vector_sku, vector_mismo, vector_edad, self.matriz_candidatas.reshape(-1).astype(float), vector_cercania,
                               vector_tarea, [escalar_balance], vector_perfil_entran, vector_perfil_salen]).astype(np.float32)

    def action_masks(self):
        return self.matriz_candidatas.reshape(-1).copy()

    def step(self, escalar_accion):
        if self.terminado:
            raise RuntimeError("El episodio terminó: llama a reset().")
        self.escalar_decisiones += 1
        self.escalar_penalizacion = 0.0
        self.cola_accion.put(int(escalar_accion))
        self._esperar_decision()
        escalar_recompensa = -(self.G.escalar_reloj - self.escalar_t_anterior) / ESCALAR_SEGUNDOS_POR_UNIDAD - self.escalar_penalizacion
        self.escalar_t_anterior = self.G.escalar_reloj
        dic_info = {"tiempo_total": self.tiempo_total()} if self.terminado else {}      # al terminar se informa la métrica de comparación
        return self._observacion(), escalar_recompensa, self.terminado, False, dic_info

    def tiempo_total(self):
        # Tiempo total de operación de la lista, la misma métrica que reportan las políticas clásicas
        return float(self.C.instancia_metricas.recuperarInformcion()[-1])


def crear_entorno_monitoreado(config, escalar_semilla_base, escalar_estados, escalar_semilla_episodios):
    # Función de nivel superior (no una lambda) para que se pueda enviar a los procesos de los entornos paralelos
    from stable_baselines3.common.monitor import Monitor
    return Monitor(EntornoAsignacion(config, escalar_semilla_base, escalar_estados, escalar_semilla_episodios=escalar_semilla_episodios))
