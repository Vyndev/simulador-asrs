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
from asrs.generacion.secuencia import GeneradorSecuencia
from asrs.politicas.recuperacion import PoliticaRecuperacionFCFS
from asrs.politicas.registro import crear_politica
from asrs.simulacion.resultados import guardar_resultados_csv


class Simulacion:
    def __init__(self, config = None):
        # Se contruye la instancia de simulación con la configuración fijada
        self.config = config if config is not None else cargar_configuracion()

    def crearInstancias(self):
        # Se crean la estantería, la grúa y el punto E/S vacíos
        cfg = self.config

        # De forma intuitiva se establece una separación entre el riel y la cara frontal de la estantería
        vector_coordenada_z_riel_grua = cfg["escalar_separacion_riel_grua"] + (cfg["vector_configuracion_estanteria"][2] * cfg["vector_cotas_estanteria"][2])

        # A fin de simplificar el modelo se establece que el punto de referencia central o centro del espacio es el punto ES
        vector_posicion_puntoES = np.array([0, 0, vector_coordenada_z_riel_grua], dtype=float)

        # A fin de simplificar el modelo se establece que el punto de reposo de la grúa es la misma que la del puntoES
        vector_posicion_reposo_grua = np.array([0, 0, vector_coordenada_z_riel_grua], dtype=float)

        E = Estanteria(cfg["vector_posicion_estanteria"], cfg["vector_configuracion_estanteria"], cfg["vector_cotas_estanteria"], cfg["vector_centro_celda_estanteria"])
        G = Grua(vector_posicion_reposo_grua, cfg["vector_velocidad_grua"], cfg["escalar_tiempo_carga_descarga"])
        P = PuntoES(vector_posicion_puntoES)

        return E, G, P

    def generarSecuencias(self):
        # Se generan la secuencia de llenado y la secuencia de operación, comunes a todas las políticas de la réplica
        cfg = self.config

        # Las llegadas y las solicitudes de cada SKU siguen su demanda, las tareas no saben nada de clases
        vector_prob_acum_sku = np.cumsum(cfg["vector_demanda_sku"])
        vector_prob_acum_sku[-1] = 1.0

        # Se calcula la cantidad de almacenamientos a realizar en el llenado, con el fin de alcanzar el nivel de ocupación fijado
        escalar_capacidad_total = int(np.prod(cfg["vector_configuracion_estanteria"]))
        escalar_largo_secuencia_llenado = int(escalar_capacidad_total * cfg["escalar_nivel_ocupacion"])

        SA = GeneradorSecuencia(escalar_largo_secuencia_llenado, np.array([1, 1]), vector_prob_acum_sku).generarSecuencia()
        S = GeneradorSecuencia(cfg["escalar_largo_secuencia"], cfg["vector_prob_acum_AR"], vector_prob_acum_sku).generarSecuencia()

        return SA, S

    def prepararEscenario(self, txt_clave_politica, SA):
        # Crea las instancias de una política y las llena con la secuencia de llenado, las métricas del llenado se descartan

        # Cada escenario cuenta las etiquetas de generación desde cero
        Item.escalar_contador_global_generacion = 0

        E, G, P = self.crearInstancias()
        instancia_politica = crear_politica(txt_clave_politica, E, G, P, self.config)
        instancia_politica_recuperacion = PoliticaRecuperacionFCFS()

        # Se ejecuta la secuencia de llenado al nivel de ocupación fijado
        ControladorGrua(E, G, P, SA, instancia_politica, instancia_politica_recuperacion).ejecutarSecuencia()

        return E, G, P, instancia_politica, instancia_politica_recuperacion

    def ejecutarEscenario(self, txt_clave_politica, SA, S):
        # Ejecuta una política: llena la estantería con ella y luego procesa la secuencia de operación
        E, G, P, instancia_politica, instancia_politica_recuperacion = self.prepararEscenario(txt_clave_politica, SA)

        # Se ejecuta la secuencia de operación sobre el estado alcanzado
        C = ControladorGrua(E, G, P, S, instancia_politica, instancia_politica_recuperacion)
        C.ejecutarSecuencia()

        # Se recuperan las métricas resultado de la simulación y, con fin de llevar trazabilidad, se incluye la política utilizada
        vector_resultados_escenario = C.instancia_metricas.recuperarInformcion()
        vector_resultados_escenario.insert(0, instancia_politica.nombre)
        return vector_resultados_escenario

    @staticmethod
    def cargarReplica(vector_argumentos_replica):
        # Ejecuta una réplica completa: secuencias propias con semilla y todas las políticas sobre esas mismas secuencias
        config, escalar_semilla = vector_argumentos_replica
        instancia_simulacion = Simulacion(config)

        rnd.seed(escalar_semilla)
        SA, S = instancia_simulacion.generarSecuencias()

        vector_resultados_replica = []
        for escalar_indice_politica, txt_clave_politica in enumerate(config["vector_politicas_asignacion"]):
            # Cada política tiene su propia semilla derivada, así el resultado no depende del orden de ejecución
            rnd.seed(escalar_semilla * 1000 + escalar_indice_politica)
            vector_resultados_replica.append(instancia_simulacion.ejecutarEscenario(txt_clave_politica, SA, S))

        return vector_resultados_replica

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
            resultados_por_replica = bahia.map(Simulacion.cargarReplica, matriz_lista_replicas)

        resultados = [vector_resultado for resultados_replica in resultados_por_replica for vector_resultado in resultados_replica]
        guardar_resultados_csv(resultados)
