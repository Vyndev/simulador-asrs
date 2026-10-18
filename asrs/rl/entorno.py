"""Entorno de aprendizaje por refuerzo para la asignación de almacenamiento.

El agente decide en qué canal almacenar cada ítem de la lista de tareas. Dentro del canal se aplica la regla de minimización
de varianza (la celda libre y accesible más profunda), igual que en las demás políticas. Las recuperaciones son FCFS y las
reubicaciones las decide una política clásica (por defecto la más cercana a la grúa), así el agente aprende solo la decisión de almacenar.

El episodio parte de una estantería ya llena (con una política de llenado neutra) y recorre la lista de tareas.
La recompensa de cada decisión es el tiempo que la grúa invierte desde esa decisión hasta la siguiente (con signo negativo, en minutos),
así que el retorno del episodio es el tiempo total de la lista, que es la métrica con la que se comparan las políticas.
"""
import copy
import random as rnd

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from asrs.control.controlador_grua import ControladorGrua
from asrs.dominio.item import Item
from asrs.politicas.recuperacion import PoliticaRecuperacionFCFS
from asrs.politicas.registro import crear_politica
from asrs.politicas.restriccion import MatricesRestriccion
from asrs.simulacion.simulacion import Simulacion

ESCALAR_SEGUNDOS_POR_UNIDAD = 60.0          # la recompensa se mide en minutos
ESCALAR_PENALIZACION_INVALIDA = 1.0         # recompensa negativa por elegir un canal sin celda disponible


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


class EntornoAsignacion(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config, escalar_semilla_base=0, escalar_estados=16, txt_politica_llenado='aleatoria', txt_politica_reubicacion='mas_cercana_grua'):
        super().__init__()
        self.config = config
        self.escalar_semilla_base = escalar_semilla_base
        self.escalar_estados = escalar_estados
        self.txt_politica_llenado = txt_politica_llenado
        self.txt_politica_reubicacion = txt_politica_reubicacion
        self.dic_estados = {}                       # estados iniciales ya generados, se crean la primera vez que se usan
        self.generador = np.random.default_rng(escalar_semilla_base)

        escalar_filas, escalar_columnas, escalar_capas = (int(x) for x in config["vector_configuracion_estanteria"])
        self.escalar_filas, self.escalar_columnas, self.escalar_capas = escalar_filas, escalar_columnas, escalar_capas
        self.escalar_canales = escalar_filas * escalar_columnas
        self.escalar_skus = int(config["escalar_numero_skus"])

        # Rango de cada SKU por demanda (0 es el más demandado), el agente lo ve normalizado
        vector_orden = np.argsort(-np.asarray(config["vector_demanda_sku"]), kind='stable')
        self.vector_rango_sku = np.empty(self.escalar_skus)
        self.vector_rango_sku[vector_orden] = np.arange(self.escalar_skus)

        # Observación: por canal (llenado, cercanía al E/S, ítems del mismo SKU, tiene celda disponible) y dos escalares
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4 * self.escalar_canales + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.escalar_canales)

        self.E = self.G = self.P = self.S = self.C = None
        self.escalar_indice = 0
        self.escalar_sku_actual = 0
        self.vector_valido = np.zeros(self.escalar_canales, dtype=bool)
        self.matriz_profundidad = np.zeros((escalar_filas, escalar_columnas), dtype=int)
        self.escalar_decisiones = 0
        self.escalar_invalidas = 0

    # ---- ciclo de vida ----
    def _estado(self, escalar_indice):
        if escalar_indice not in self.dic_estados:
            self.dic_estados[escalar_indice] = generar_estado(self.config, self.escalar_semilla_base + escalar_indice, self.txt_politica_llenado)
        return self.dic_estados[escalar_indice]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Con options={'estado': i} se elige un estado concreto, si no se sortea uno del conjunto
        escalar_indice = options["estado"] if options and "estado" in options else int(self.generador.integers(self.escalar_estados))
        self.E, self.G, self.P, self.S = copiar_estado(self._estado(escalar_indice))
        rnd.seed(self.escalar_semilla_base * 7919 + escalar_indice)
        politica_reubicacion = crear_politica(self.txt_politica_reubicacion, self.E, self.G, self.P, self.config)
        self.C = ControladorGrua(self.E, self.G, self.P, self.S, politica_reubicacion, PoliticaRecuperacionFCFS())
        self.escalar_indice = 0
        self.escalar_decisiones = 0
        self.escalar_invalidas = 0
        self._avanzar()
        return self._observacion(), {}

    # ---- mecánica ----
    def _avanzar(self):
        # Ejecuta las tareas hasta la próxima decisión de almacenamiento, devuelve False si la lista terminó
        while self.escalar_indice < len(self.S):
            escalar_actividad, escalar_sku = self.S[self.escalar_indice]
            if escalar_actividad == 1:
                vector_indice = self.C.instancia_politica_recuperacion.seleccionar(self.E, escalar_sku)
                if len(vector_indice):
                    self.C.recuperar(vector_indice)
                self.escalar_indice += 1
                continue
            self._calcular_candidatas()
            if self.vector_valido.any():
                self.escalar_sku_actual = escalar_sku
                return True
            self.escalar_indice += 1                 # sin espacio, la tarea se omite como en las demás políticas
        return False

    def _calcular_candidatas(self):
        # Celda libre y accesible más profunda de cada canal
        matriz_ocupacion = MatricesRestriccion().ocupacion(self.E)
        matriz_candidatas = ~matriz_ocupacion & MatricesRestriccion().accesibilidad(matriz_ocupacion)
        self.matriz_profundidad = np.argmax(matriz_candidatas, axis=2)
        self.vector_valido = matriz_candidatas.any(axis=2).reshape(-1)
        self.matriz_ocupacion = matriz_ocupacion

    def action_masks(self):
        return self.vector_valido.copy()

    def _observacion(self):
        if self.escalar_indice >= len(self.S):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        escalar_capas = self.escalar_capas
        matriz_llenado = self.matriz_ocupacion.sum(axis=2) / escalar_capas
        # Cercanía al E/S: tiempo de viaje desde el E/S hasta cada canal, normalizado (1 = el más cercano)
        politica = self.C.instancia_politica_asignacion
        matriz_tiempos = politica.tiempoChebyshevXY(self.P.vector_posicion)[:, :, 0]
        matriz_cercania = 1.0 - matriz_tiempos / max(float(matriz_tiempos.max()), 1e-9)
        # Cantidad de ítems del mismo SKU que ya hay en cada canal
        matriz_mismo_sku = np.zeros((self.escalar_filas, self.escalar_columnas))
        for f, c, k in np.argwhere(self.matriz_ocupacion):
            if self.E.matriz_contenedor_items[f, c, k].escalar_sku == self.escalar_sku_actual:
                matriz_mismo_sku[f, c] += 1
        matriz_mismo_sku /= escalar_capas
        escalar_rango = self.vector_rango_sku[self.escalar_sku_actual - 1] / max(self.escalar_skus - 1, 1)
        escalar_avance = self.escalar_indice / len(self.S)
        return np.concatenate([matriz_llenado.reshape(-1), matriz_cercania.reshape(-1), matriz_mismo_sku.reshape(-1),
                               self.vector_valido.astype(float), [escalar_rango, escalar_avance]]).astype(np.float32)

    def step(self, escalar_accion):
        escalar_accion = int(escalar_accion)
        escalar_penalizacion = 0.0
        self.escalar_decisiones += 1
        if not self.vector_valido[escalar_accion]:
            # Canal sin celda disponible: se usa el válido más cercano al E/S y se penaliza
            self.escalar_invalidas += 1
            escalar_penalizacion = ESCALAR_PENALIZACION_INVALIDA
            matriz_tiempos = self.C.instancia_politica_asignacion.tiempoChebyshevXY(self.P.vector_posicion)[:, :, 0].reshape(-1)
            escalar_accion = int(np.argmin(np.where(self.vector_valido, matriz_tiempos, np.inf)))
        escalar_fila, escalar_columna = divmod(escalar_accion, self.escalar_columnas)
        tupla_celda = (escalar_fila, escalar_columna, int(self.matriz_profundidad[escalar_fila, escalar_columna]))

        escalar_t0 = self.G.escalar_reloj
        self.P.matriz_contenedor_temporal_items.append(Item(self.escalar_sku_actual))
        self.C.almacenar(tupla_celda)
        self.escalar_indice += 1
        booleano_hay_mas = self._avanzar()
        escalar_recompensa = -(self.G.escalar_reloj - escalar_t0) / ESCALAR_SEGUNDOS_POR_UNIDAD - escalar_penalizacion
        return self._observacion(), escalar_recompensa, not booleano_hay_mas, False, {}

    def tiempo_total(self):
        # Tiempo total de operación de la lista, la misma métrica que reportan las políticas clásicas
        return float(self.C.instancia_metricas.recuperarInformcion()[-1])
