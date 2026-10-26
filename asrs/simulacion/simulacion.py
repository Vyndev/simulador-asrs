"""Orquestación de réplicas y escenarios de simulación."""
import multiprocessing as mp
import random as rnd

import numpy as np

from asrs.config import cargar_configuracion
from asrs.control.controlador_grua import ControladorGrua
from asrs.dominio.estanteria import Estanteria
from asrs.dominio.grua import Grua
from asrs.dominio.item import Item
from asrs.dominio.punto_es import PuntoES
from asrs.generacion.probabilidades import perfil_inventario
from asrs.generacion.reposicion import generar_escenario
from asrs.generacion.secuencia import GeneradorSecuencia
from asrs.politicas.registro import POLITICAS_ASIGNACION, crear_politica, crear_politica_recuperacion
from asrs.simulacion.resultados import guardar_resultados_csv, guardar_resumen_escenarios_csv


class Simulacion:
    def __init__(self, config = None):
        # Se contruye la instancia de simulación con la configuración fijada
        self.config = config if config is not None else cargar_configuracion()
        # Escenario de demanda de la última réplica generada, None con datos reales y antes de generar las secuencias
        self.escenario_demanda = None

    def configPolitica(self):
        # Configuración que reciben las políticas: la misma de la simulación más el escenario, del que leen la popularidad móvil
        return dict(self.config, escenario_demanda=self.escenario_demanda)

    def argumentosOperacion(self, escalar_tareas):
        # Argumentos del controlador para ejecutar las primeras tareas de la lista de operación: el día de cada tarea y los retiros
        # Sin escenario (datos reales) no hay avisos de día ni retiros
        if self.escenario_demanda is None:
            return {}
        return {"vector_dia_de_tarea": self.escenario_demanda.vector_dia_de_tarea[:escalar_tareas],
                "conjunto_retiros": {escalar_indice for escalar_indice in self.escenario_demanda.conjunto_retiros if escalar_indice < escalar_tareas}}

    def crearInstancias(self):
        # Se crean la estantería, la grúa y el punto E/S vacíos
        cfg = self.config

        # De forma intuitiva se establece una separación entre el riel y la cara frontal de la estantería
        # La cara frontal está en la posición de profundidad de la estantería más su profundidad total, así el riel queda siempre delante
        vector_coordenada_z_riel_grua = (cfg["vector_posicion_estanteria"][2] + cfg["vector_configuracion_estanteria"][2] * cfg["vector_cotas_estanteria"][2]
                                         + cfg["escalar_separacion_riel_grua"])

        # A fin de simplificar el modelo se establece que el punto de referencia central o centro del espacio es el punto ES
        vector_posicion_puntoES = np.array([0, 0, vector_coordenada_z_riel_grua], dtype=float)

        # A fin de simplificar el modelo se establece que el punto de reposo de la grúa es la misma que la del puntoES
        vector_posicion_reposo_grua = np.array([0, 0, vector_coordenada_z_riel_grua], dtype=float)

        E = Estanteria(cfg["vector_posicion_estanteria"], cfg["vector_configuracion_estanteria"], cfg["vector_cotas_estanteria"], cfg["vector_centro_celda_estanteria"])
        G = Grua(vector_posicion_reposo_grua, cfg["vector_velocidad_grua"], cfg["escalar_tiempo_toma_deposito_es"],
                 cfg.get("vector_aceleracion_grua"), cfg.get("vector_deceleracion_grua"),
                 cfg["escalar_tiempo_toma_deposito_capa_delantera"], cfg["escalar_tiempo_toma_deposito_capa_trasera"])
        P = PuntoES(vector_posicion_puntoES)

        return E, G, P

    def generarSecuencias(self):
        # Se generan la secuencia de llenado y la secuencia de operación, comunes a todas las políticas de la réplica
        cfg = self.config

        if cfg.get("texto_origen_tareas") == "datos":
            # Lista real: es la misma en todas las réplicas, solo cambia el llenado inicial
            self.escenario_demanda = None

            # El llenado inicial sigue el perfil de inventario: cuántas unidades de cada SKU hay guardadas (no cuántas veces se piden)
            vector_prob_acum_inventario = np.cumsum(perfil_inventario(cfg["vector_demanda_sku"], cfg["texto_perfil_inventario"]))
            vector_prob_acum_inventario[-1] = 1.0

            # Se calcula la cantidad de almacenamientos a realizar en el llenado, con el fin de alcanzar el nivel de ocupación fijado
            escalar_capacidad_total = int(np.prod(cfg["vector_configuracion_estanteria"]))
            escalar_largo_secuencia_llenado = int(escalar_capacidad_total * cfg["escalar_nivel_ocupacion"])

            SA = GeneradorSecuencia(escalar_largo_secuencia_llenado, np.array([1, 1]), vector_prob_acum_inventario).generarSecuencia()
            S = [(int(escalar_actividad), int(escalar_sku)) for escalar_actividad, escalar_sku in cfg["secuencia_real"]]
            return SA, S

        # Lista generada: el modelo de suministro y demanda entrega el llenado (inventario al final del calentamiento) y la operación
        # Todo su azar sale de un generador propio, sembrado desde la semilla de la réplica
        self.escenario_demanda = generar_escenario(cfg, np.random.default_rng(rnd.getrandbits(64)))
        return self.escenario_demanda.lista_llenado, self.escenario_demanda.lista_operacion

    @staticmethod
    def indiceSemillaPolitica(txt_clave_politica):
        # Índice que fija el azar de una política: el de su política base en el registro, así no depende de qué otras políticas se eligen
        # Una variante replanificada comparte el índice de la de plan fijo: parten del mismo plan y sus desempates al azar son los mismos
        # Una clave desconocida recibe el índice siguiente al último, crear_politica la rechaza con un mensaje claro
        vector_claves = list(POLITICAS_ASIGNACION)
        txt_base = txt_clave_politica.removesuffix('_replanificada')
        return vector_claves.index(txt_base) if txt_base in vector_claves else len(vector_claves)

    @staticmethod
    def semillaDePolitica(escalar_semilla, txt_clave_politica):
        # Semilla con la que se ejecuta una política dentro de una réplica
        return escalar_semilla * 1000 + Simulacion.indiceSemillaPolitica(txt_clave_politica)

    @staticmethod
    def escenarios(config):
        # Combinaciones de política de asignación y de recuperación que se comparan en cada réplica
        # Devuelve (índice de la política de asignación en el registro, clave de asignación, clave de recuperación)
        # El índice fija la semilla: las dos recuperaciones de una misma asignación comparten el azar y la comparación entre ellas es pareada
        return [(Simulacion.indiceSemillaPolitica(txt_clave_politica), txt_clave_politica, txt_clave_recuperacion)
                for txt_clave_politica in config["vector_politicas_asignacion"]
                for txt_clave_recuperacion in config.get("vector_politicas_recuperacion", ('fcfs',))]

    def prepararEscenario(self, txt_clave_politica, SA, txt_clave_recuperacion = 'fcfs'):
        # Crea las instancias de una política y las llena con la secuencia de llenado, las métricas del llenado se descartan

        # Cada escenario cuenta las etiquetas de generación desde cero
        Item.escalar_contador_global_generacion = 0

        E, G, P = self.crearInstancias()
        instancia_politica = crear_politica(txt_clave_politica, E, G, P, self.configPolitica())
        instancia_politica_recuperacion = crear_politica_recuperacion(txt_clave_recuperacion)

        # Se ejecuta la secuencia de llenado al nivel de ocupación fijado
        ControladorGrua(E, G, P, SA, instancia_politica, instancia_politica_recuperacion).ejecutarSecuencia()

        return E, G, P, instancia_politica, instancia_politica_recuperacion

    def ejecutarEscenario(self, txt_clave_politica, SA, S, txt_clave_recuperacion = 'fcfs'):
        # Ejecuta una política: llena la estantería con ella y luego procesa la secuencia de operación
        E, G, P, instancia_politica, instancia_politica_recuperacion = self.prepararEscenario(txt_clave_politica, SA, txt_clave_recuperacion)

        # Se ejecuta la secuencia de operación sobre el estado alcanzado
        C = ControladorGrua(E, G, P, S, instancia_politica, instancia_politica_recuperacion, **self.argumentosOperacion(len(S)))
        C.ejecutarSecuencia()

        # Se recuperan las métricas resultado de la simulación y, con fin de llevar trazabilidad, se incluyen las políticas utilizadas
        vector_resultados_escenario = C.instancia_metricas.recuperarInformcion()
        vector_resultados_escenario[0:0] = [instancia_politica.nombre, instancia_politica_recuperacion.nombre]
        return vector_resultados_escenario

    @staticmethod
    def cargarReplicaConEscenario(vector_argumentos_replica):
        # Ejecuta una réplica completa: secuencias propias con semilla y todas las políticas sobre esas mismas secuencias
        # Devuelve las filas de resultados y el resumen del escenario de la réplica (con datos reales, solo la semilla)
        config, escalar_semilla = vector_argumentos_replica
        instancia_simulacion = Simulacion(config)

        rnd.seed(escalar_semilla)
        SA, S = instancia_simulacion.generarSecuencias()

        vector_resultados_replica = []
        for _, txt_clave_politica, txt_clave_recuperacion in Simulacion.escenarios(config):
            # Cada política tiene su propia semilla derivada, así el resultado no depende del orden de ejecución ni de las demás políticas elegidas
            rnd.seed(Simulacion.semillaDePolitica(escalar_semilla, txt_clave_politica))
            vector_resultados_replica.append(instancia_simulacion.ejecutarEscenario(txt_clave_politica, SA, S, txt_clave_recuperacion))

        dic_resumen = {"semilla": escalar_semilla}
        if instancia_simulacion.escenario_demanda is not None:
            dic_resumen.update(instancia_simulacion.escenario_demanda.resumen)
        return vector_resultados_replica, dic_resumen

    @staticmethod
    def cargarReplica(vector_argumentos_replica):
        # Solo las filas de resultados de la réplica, el resumen del escenario se descarta
        return Simulacion.cargarReplicaConEscenario(vector_argumentos_replica)[0]

    def ejecutarSimulacion(self):

        # en esta función se ingran las funciones anteriores
        # Se ejecuta la simulación usando procesamiento en paralelo a fin de reducir el tiempo de computo,
        cfg = self.config

        # Cada réplica es independiente (secuencias propias) y reproducible (semilla propia)
        matriz_lista_replicas = [(cfg, cfg["escalar_semilla"] + escalar_replica) for escalar_replica in range(cfg["escalar_cantidad_replicas"])]

        # Se crea una bahía de procesos
        # Si se deja 'processes=mp.cpu_count()' entonces usará todos los núcleos
        # En cambio si se remplaza 'mo.cpu_count()' por algún número entero menor que la cantidad de núcleos disponibles en el procesador, entonces se limitara la cantidad de núcleos utilizados
        with mp.Pool(processes=mp.cpu_count()) as bahia:
            resultados_por_replica = bahia.map(Simulacion.cargarReplicaConEscenario, matriz_lista_replicas)

        resultados = [vector_resultado for resultados_replica, _ in resultados_por_replica for vector_resultado in resultados_replica]
        guardar_resultados_csv(resultados)
        guardar_resumen_escenarios_csv([dic_resumen for _, dic_resumen in resultados_por_replica])
