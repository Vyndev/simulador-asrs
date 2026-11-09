"""Prototipo: entorno Gymnasium del agente de asignación (capítulo 9), común a las dos opciones que se comparan.

Una decisión por paso: cada vez que el controlador pide una celda (almacenar o reubicar), la simulación se detiene y espera la acción.
El control se invierte con un hilo: el controlador corre en un hilo aparte y la política del agente le pasa al entorno el estado y
espera la acción. El intercambio es estricto (un solo lado avanza a la vez), así que la ejecución es determinista.

Observación (vector de tamaño fijo):
  estado s_t (7): perfil del SKU por ubicar (nivel, tendencia, historia), fracción ocupada, posición de la grúa (fila y columna
                  normalizadas) y 1 si la decisión es una reubicación.
  candidatas (N x 5): por cada celda de la estantería, las características de `celdas_candidatas` (tiempo de viaje, capa y perfil de la
                  unidad de atrás); ceros si la celda no es candidata.
Acción: índice de la celda (Discrete(N)); `action_masks()` marca las candidatas.
Recompensa: -(ΔT + PENALIZACION_PERDIDA · pérdidas) / ESCALA_RECOMPENSA, con ΔT el tiempo de la grúa entre esta decisión y la siguiente
(ec. 9.9) y pérdidas las tareas que no se pudieron ejecutar en ese intervalo (almacenamientos sin celda accesible y recuperaciones sin
unidad). Sin la penalización, un agente libre reduce el tiempo dejando fondos inaccesibles: descarta almacenamientos y luego no
encuentra las unidades, y hace menos operaciones (hallazgo del prototipo).

Acción: el agente aplica la regla de minimización de varianza, como las heurísticas (`mascara_candidatas`): elige entre las candidatas de
la capa libre más profunda, así nunca deja un fondo inaccesible y no pierde tareas. La penalización de pérdidas queda como salvaguarda.

Simplificación del prototipo: el llenado inicial lo hace la política aleatoria (con la regla de varianza) para todas las políticas,
de modo que todas parten del mismo estado; la operación la ejecuta cada política.
"""
import queue
import random as rnd
import threading

import gymnasium as gym
import numpy as np

from asrs.agente.candidatos import celdas_candidatas
from asrs.agente.perfil import perfil_de_skus
from asrs.config import cargar_configuracion, config_a_json, config_desde_json
from asrs.control.controlador_grua import ControladorGrua
from asrs.politicas.asignacion import PoliticaAsignacion
from asrs.politicas.registro import crear_politica, crear_politica_recuperacion
from asrs.simulacion.simulacion import Simulacion

ESCALA_RECOMPENSA = 100.0  # segundos: la recompensa queda del orden de 0,1 a 1
PENALIZACION_PERDIDA = 300.0  # segundos por tarea perdida, unas diez operaciones medias (34 s cada una en la instancia del prototipo)
DIM_ESTADO = 7
DIM_CANDIDATA = 5


def config_prototipo(escalar_tareas=1000, vector_estanteria=(6, 6, 2), escalar_skus=20):
    datos = config_a_json(cargar_configuracion())
    datos.update(vector_configuracion_estanteria=list(vector_estanteria), escalar_numero_skus=escalar_skus, escalar_largo_secuencia=escalar_tareas,
                 vector_politicas_asignacion=["aleatoria"], vector_politicas_recuperacion=["fcfs"])
    return config_desde_json(datos)


def escenario_lleno(config, escalar_semilla, txt_politica_llenado="aleatoria"):
    """Genera la réplica de la semilla y llena la estantería con la política de llenado. Devuelve (simulación, E, G, P, S)."""
    simulacion = Simulacion(config)
    rnd.seed(escalar_semilla)
    SA, S = simulacion.generarSecuencias(escalar_semilla)
    rnd.seed(escalar_semilla * 1000 + 7)
    E, G, P, _, _ = simulacion.prepararEscenario(txt_politica_llenado, SA)
    return simulacion, E, G, P, S


class _Abortar(Exception):
    pass


class PoliticaAgente(PoliticaAsignacion):
    """Política que delega cada decisión en `decidir(observación, máscara) -> índice de celda`."""
    nombre = "agente"

    def __init__(self, E, G, P, config, decidir):
        super().__init__(E, G, P, config)
        self.decidir = decidir
        self.escenario = config["escenario_demanda"]
        self.escalar_ventana = config["escalar_dias_ventana_popularidad"]
        self.forma = E.matriz_contenedor_items.shape
        self.escalar_celdas = int(np.prod(self.forma))
        self.matriz_perfil = perfil_de_skus(self.escenario, 0, self.escalar_ventana)
        self.vector_extension = np.maximum(np.array(self.forma[:2], dtype=float) * np.asarray(E.vector_cotas[:2], dtype=float), 1e-9)

    def iniciarDia(self, escalar_dia):
        # El perfil solo cambia de un día a otro
        self.matriz_perfil = perfil_de_skus(self.escenario, escalar_dia, self.escalar_ventana)

    def matrizCriterio(self, vector_posicion_referencia=None):
        raise NotImplementedError

    def observar(self, escalar_sku, matriz_extra, vector_posicion_referencia):
        vector_referencia = self.instancia_puntoES.vector_posicion if vector_posicion_referencia is None else vector_posicion_referencia
        matriz_tiempos = self.tiempoChebyshevXY(vector_referencia)
        # Las candidatas ya aplican la restricción de la reubicación y la regla de minimización de varianza
        matriz_indices, matriz_caract = celdas_candidatas(self.instancia_estanteria, matriz_tiempos, self.matriz_perfil, matriz_extra)
        vector_plano = np.ravel_multi_index(tuple(matriz_indices.T), self.forma) if len(matriz_indices) else np.zeros(0, dtype=int)
        matriz_celdas = np.zeros((self.escalar_celdas, DIM_CANDIDATA), dtype=np.float32)
        matriz_celdas[vector_plano] = matriz_caract
        vector_mascara = np.zeros(self.escalar_celdas, dtype=bool)
        vector_mascara[vector_plano] = True
        vector_perfil = self.matriz_perfil[escalar_sku - 1] if escalar_sku <= len(self.matriz_perfil) else np.zeros(3)
        escalar_ocupacion = float(np.count_nonzero(self.instancia_estanteria.matriz_contenedor_items != 0)) / self.escalar_celdas
        vector_grua = (np.asarray(self.instancia_grua.vector_posicion_actual[:2], dtype=float)
                       - np.asarray(self.instancia_estanteria.vector_posicion[:2], dtype=float)) / self.vector_extension
        vector_estado = np.concatenate([vector_perfil, [escalar_ocupacion], np.clip(vector_grua, -1, 2),
                                        [1.0 if vector_posicion_referencia is not None else 0.0]]).astype(np.float32)
        return np.concatenate([vector_estado, matriz_celdas.ravel()]), vector_mascara

    def asignar(self, escalar_sku, matriz_extra=None, booleano_restringir_zona=True, vector_posicion_referencia=None):
        vector_obs, vector_mascara = self.observar(escalar_sku, matriz_extra, vector_posicion_referencia)
        if not vector_mascara.any():
            return ()
        escalar_accion = int(self.decidir(vector_obs, vector_mascara))
        if not vector_mascara[escalar_accion]:
            raise ValueError(f"Acción {escalar_accion} no es una candidata.")
        return tuple(int(v) for v in np.unravel_index(escalar_accion, self.forma))


class EntornoAsignacion(gym.Env):
    """Un episodio es la lista de operación de una réplica, con la semilla de `vector_semillas` (en ciclo)."""
    metadata = {"render_modes": []}

    def __init__(self, config, vector_semillas):
        self.config = config
        self.vector_semillas = list(vector_semillas)
        self.escalar_episodio = 0
        self.escalar_celdas = int(np.prod(config["vector_configuracion_estanteria"]))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(DIM_ESTADO + self.escalar_celdas * DIM_CANDIDATA,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.escalar_celdas)
        self.hilo = None
        self.vector_mascara = np.ones(self.escalar_celdas, dtype=bool)
        self.dic_ultimo = {}

    # --- lado del hilo del simulador
    def _decidir(self, vector_obs, vector_mascara):
        self.cola_salida.put(("decision", vector_obs, vector_mascara, self.G.escalar_reloj, self._perdidas()))
        escalar_accion = self.cola_entrada.get()
        if escalar_accion is None:
            raise _Abortar()
        return escalar_accion

    def _correr(self, controlador):
        try:
            controlador.ejecutarSecuencia()
            self.cola_salida.put(("fin", controlador.instancia_metricas.recuperarInformcion(), None, self.G.escalar_reloj, self._perdidas()))
        except _Abortar:
            pass
        except Exception as error:  # se propaga al hilo principal
            self.cola_salida.put(("error", error, None, None, 0))

    def _perdidas(self):
        # Tareas no ejecutadas desde la última consulta: el controlador las registra como evento 'T' con la marca 0
        escalar = sum(1 for evento in self.G.vector_eventos if evento[1] == 'T' and evento[-1] == 0)
        self.G.vector_eventos.clear()
        return escalar

    def _abortar(self):
        if self.hilo is not None and self.hilo.is_alive():
            self.cola_entrada.put(None)
            self.hilo.join()
        self.hilo = None

    # --- interfaz Gymnasium
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._abortar()
        escalar_semilla = self.vector_semillas[self.escalar_episodio % len(self.vector_semillas)]
        self.escalar_episodio += 1
        simulacion, E, G, P, S = escenario_lleno(self.config, escalar_semilla)
        self.G = G
        G.vector_eventos = []
        self.escalar_perdidas = 0
        agente = PoliticaAgente(E, G, P, simulacion.configPolitica(), self._decidir)
        controlador = ControladorGrua(E, G, P, S, agente, crear_politica_recuperacion("fcfs"), **simulacion.argumentosOperacion(len(S)))
        self.cola_entrada, self.cola_salida = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.escalar_reloj_previo = G.escalar_reloj
        self.escalar_decisiones = 0
        self.hilo = threading.Thread(target=self._correr, args=(controlador,), daemon=True)
        self.hilo.start()
        vector_obs, _, _, _ = self._esperar()
        return vector_obs, {"semilla": escalar_semilla}

    def _esperar(self):
        txt_tipo, dato, vector_mascara, escalar_reloj, escalar_perdidas = self.cola_salida.get()
        if txt_tipo == "error":
            raise dato
        self.escalar_perdidas += escalar_perdidas
        escalar_recompensa = -(escalar_reloj - self.escalar_reloj_previo + PENALIZACION_PERDIDA * escalar_perdidas) / ESCALA_RECOMPENSA
        self.escalar_reloj_previo = escalar_reloj
        if txt_tipo == "fin":
            self.dic_ultimo = {"tiempo_total": float(dato[-1]), "decisiones": self.escalar_decisiones, "perdidas": self.escalar_perdidas}
            return np.zeros(self.observation_space.shape, dtype=np.float32), escalar_recompensa, True, dict(self.dic_ultimo)
        self.vector_mascara = vector_mascara
        return dato, escalar_recompensa, False, {}

    def step(self, accion):
        self.escalar_decisiones += 1
        self.cola_entrada.put(int(accion))
        vector_obs, escalar_recompensa, booleano_fin, dic_info = self._esperar()
        return vector_obs, escalar_recompensa, booleano_fin, False, dic_info

    def action_masks(self):
        return self.vector_mascara.copy()

    def close(self):
        self._abortar()


def _resultado(controlador, S):
    # (tiempo total de operación, tareas perdidas): las tareas de la lista que no se ejecutaron
    vector = controlador.instancia_metricas.recuperarInformcion()
    return float(vector[-1]), len(S) - int(vector[-2])


def evaluar_heuristica(config, txt_politica, escalar_semilla):
    """(Tiempo total, tareas perdidas) de una heurística en la réplica, con el mismo llenado que el agente."""
    simulacion, E, G, P, S = escenario_lleno(config, escalar_semilla)
    rnd.seed(escalar_semilla * 1000 + 11)
    politica = crear_politica(txt_politica, E, G, P, simulacion.configPolitica())
    controlador = ControladorGrua(E, G, P, S, politica, crear_politica_recuperacion("fcfs"), **simulacion.argumentosOperacion(len(S)))
    controlador.ejecutarSecuencia()
    return _resultado(controlador, S)


def evaluar_agente(config, decidir, escalar_semilla):
    """(Tiempo total, tareas perdidas) con una función `decidir(obs, máscara)`, sin hilos (la decisión se toma dentro del simulador)."""
    simulacion, E, G, P, S = escenario_lleno(config, escalar_semilla)
    agente = PoliticaAgente(E, G, P, simulacion.configPolitica(), decidir)
    controlador = ControladorGrua(E, G, P, S, agente, crear_politica_recuperacion("fcfs"), **simulacion.argumentosOperacion(len(S)))
    controlador.ejecutarSecuencia()
    return _resultado(controlador, S)
