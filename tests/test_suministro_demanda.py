"""Pruebas del modelo de suministro y demanda. Ejecutar con: python -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asrs.config import cargar_configuracion
from asrs.generacion.ciclo_vida import (DIAS_POR_ANIO, DIAS_POR_MES, CarteraProductos, crear_cartera, densidad_bass,
                                        factor_estacional, muestrear_amplitudes, muestrear_dias_pico,
                                        participacion_superior)
from asrs.generacion.probabilidades import demanda_hausman
from asrs.generacion.reposicion import calibrar_kappa, generar_escenario, lotes_eoq, recuperaciones_diarias
from asrs.politicas.registro import POLITICAS_ASIGNACION, crear_politica
from asrs.politicas.zonificacion import clasificar_skus
from asrs.simulacion.simulacion import Simulacion

# Duración del ciclo de la instancia del capítulo 6 y razón q / p de Bass
DURACION_CICLO = 18 * DIAS_POR_MES
RAZON_QP = 0.38 / 0.03
TAREAS_POR_DIA = 1440 # 60 ciclos por hora durante 24 horas
TECHO_INSTANCIA = 684 # floor(0,95 · 720)


class PruebasCicloDeVida(unittest.TestCase):
    def test_densidad_pico_simetria_y_area(self):
        # La curva de Bass reescalada: pico en L/2, simétrica, 27,1 % del pico en los extremos y área 1 - 1/r
        vector_edad = np.linspace(0, DURACION_CICLO, 20_001)[:-1]
        vector_densidad = densidad_bass(vector_edad, DURACION_CICLO, RAZON_QP)
        escalar_edad_pico = vector_edad[np.argmax(vector_densidad)]
        self.assertAlmostEqual(escalar_edad_pico, DURACION_CICLO / 2, delta=0.002 * DURACION_CICLO)
        escalar_g_cero = densidad_bass(np.array([0.0]), DURACION_CICLO, RAZON_QP)[0]
        escalar_g_pico = densidad_bass(np.array([DURACION_CICLO / 2]), DURACION_CICLO, RAZON_QP)[0]
        self.assertAlmostEqual(escalar_g_cero / escalar_g_pico, 0.2710, delta=0.001)
        escalar_g_final = densidad_bass(np.array([DURACION_CICLO - 1e-9]), DURACION_CICLO, RAZON_QP)[0]
        self.assertTrue(np.isclose(escalar_g_cero, escalar_g_final, rtol=1e-6))
        escalar_area = (getattr(np, "trapezoid", None) or np.trapz)(vector_densidad, vector_edad) # np.trapz en versiones antiguas de numpy
        self.assertAlmostEqual(escalar_area, 1 - 1 / RAZON_QP, delta=0.002)

    def test_densidad_fuera_del_ciclo_y_ciclo_infinito(self):
        vector_fuera = densidad_bass(np.array([-1.0, DURACION_CICLO]), DURACION_CICLO, RAZON_QP)
        self.assertTrue(np.array_equal(vector_fuera, [0.0, 0.0]))
        vector_infinito = densidad_bass(np.array([0.0, 500.0]), np.inf, RAZON_QP)
        self.assertTrue(np.array_equal(vector_infinito, [1.0, 1.0]))

    def test_factor_estacional_amplitud_pico_y_periodo(self):
        vector_amplitud = np.array([0.2])
        vector_dia_pico = np.array([100.0])
        vector_factor = np.array([factor_estacional(escalar_dia, vector_amplitud, vector_dia_pico)[0]
                                  for escalar_dia in range(DIAS_POR_ANIO)])
        self.assertEqual(int(np.argmax(vector_factor)), 100)
        self.assertAlmostEqual(np.log(vector_factor.max() / vector_factor.min()), 0.2, delta=1e-3)
        escalar_dia_365 = factor_estacional(365, vector_amplitud, vector_dia_pico)[0]
        self.assertAlmostEqual(escalar_dia_365, vector_factor[0], places=12)

    def test_amplitudes_siguen_los_cuantiles(self):
        config = cargar_configuracion()
        vector_amplitud = muestrear_amplitudes(np.random.default_rng(1), 200_000,
                                               config["vector_cuantiles_amplitud"],
                                               config["vector_probabilidades_amplitud"])
        vector_percentiles = np.percentile(vector_amplitud, [10, 50, 90])
        for escalar_obtenido, escalar_esperado in zip(vector_percentiles, [0.10, 0.20, 0.47]):
            self.assertAlmostEqual(escalar_obtenido, escalar_esperado, delta=0.005)
        self.assertGreaterEqual(vector_amplitud.min(), 0.05)
        self.assertLessEqual(vector_amplitud.max(), 0.67)

    def test_meses_de_pico(self):
        config = cargar_configuracion()
        vector_dia_pico = muestrear_dias_pico(np.random.default_rng(2), 200_000, config["vector_reparto_mes_pico"])
        vector_mes = np.floor(vector_dia_pico / DIAS_POR_MES).astype(int)
        escalar_invierno = np.isin(vector_mes, [11, 0, 1]).mean()
        escalar_verano = np.isin(vector_mes, [5, 6, 7]).mean()
        self.assertAlmostEqual(escalar_invierno, 0.44, delta=0.005)
        self.assertAlmostEqual(escalar_verano, 0.37, delta=0.005)

    def test_cartera_estatica_es_hausman(self):
        config = cargar_configuracion()
        config["texto_modelo_demanda"] = "estatica"
        cartera = crear_cartera(config, np.random.default_rng(4))
        vector_esperado = demanda_hausman(180, 0.4)
        self.assertTrue(np.allclose(cartera.participaciones(0), vector_esperado))
        self.assertTrue(np.allclose(cartera.participaciones(200), vector_esperado))

    def test_sucesion_aumenta_la_generacion(self):
        # Una posición con edad L - 0.5 cierra su ciclo al día siguiente y el sucesor empieza con edad 0.5
        cartera = CarteraProductos(np.array([1.0]), np.array([DURACION_CICLO - 0.5]), DURACION_CICLO, RAZON_QP,
                                   np.array([0.2]), np.array([100.0]), 30)
        self.assertEqual(cartera.generaciones(0)[0], 0)
        self.assertEqual(cartera.generaciones(1)[0], 1)
        self.assertAlmostEqual(cartera.edades(1)[0], 0.5, places=9)

    def test_calibracion_da_la_concentracion_de_sigma_04(self):
        config = cargar_configuracion()
        cartera = crear_cartera(config, np.random.default_rng(3))
        vector_concentracion = [participacion_superior(cartera.participaciones(escalar_dia))
                                for escalar_dia in range(30, 395)]
        self.assertAlmostEqual(float(np.mean(vector_concentracion)), 0.2 ** 0.4, delta=0.003)
        self.assertGreater(cartera.escalar_sigma_alturas, 0.38)
        self.assertLess(cartera.escalar_sigma_alturas, 0.48)

    def test_con_menos_de_cinco_posiciones_no_se_calibra(self):
        # El 20 % superior de 4 posiciones no contiene ninguna: se usa el σ configurado y la cartera funciona
        config = cargar_configuracion()
        config["escalar_numero_skus"] = 4
        cartera = crear_cartera(config, np.random.default_rng(3))
        self.assertEqual(cartera.escalar_sigma_alturas, config["escalar_sigma_hausman"])
        self.assertAlmostEqual(float(cartera.participaciones(30).sum()), 1.0)


def config_escenario(**dic_cambios):
    # Configuración por defecto con 60 días de operación, salvo los cambios indicados
    config = cargar_configuracion()
    config["escalar_largo_secuencia"] = 60 * TAREAS_POR_DIA
    config.update(dic_cambios)
    return config


def generar(config, escalar_semilla=11):
    return generar_escenario(config, np.random.default_rng(escalar_semilla))


def recuperaciones_por_dia(escenario, escalar_sku=None):
    # Recuperaciones de demanda (sin los retiros de fin de ciclo) de cada día de operación, de un SKU o de todos
    vector_cuenta = np.zeros(escenario.matriz_popularidad.shape[0], dtype=int)
    for escalar_indice, (escalar_actividad, escalar_sku_tarea) in enumerate(escenario.lista_operacion):
        if escalar_actividad == 1 and escalar_indice not in escenario.conjunto_retiros and escalar_sku in (None, escalar_sku_tarea):
            vector_cuenta[escenario.vector_dia_de_tarea[escalar_indice]] += 1
    return vector_cuenta


def resumenes_iguales(dic_a, dic_b):
    # Compara dos resúmenes tomando como iguales dos NaN (σ de las alturas con demanda personalizada)
    if dic_a.keys() != dic_b.keys():
        return False
    return all((isinstance(dic_a[txt], float) and isinstance(dic_b[txt], float) and np.isnan(dic_a[txt]) and np.isnan(dic_b[txt]))
               or dic_a[txt] == dic_b[txt] for txt in dic_a)


class PruebasEscenarioDemanda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # El escenario por defecto se genera una vez y lo comparten las pruebas que no cambian la configuración
        cls.config = config_escenario()
        cls.escenario = generar(cls.config)

    def comprobar_inventario(self, escenario, escalar_techo):
        # Recorre la lista de llenado y luego la de operación con un contador por SKU:
        # ninguna recuperación encuentra cero unidades y el total nunca pasa el techo
        dic_contador = {}
        for escalar_actividad, escalar_sku in escenario.lista_llenado:
            self.assertEqual(escalar_actividad, 0)
            dic_contador[escalar_sku] = dic_contador.get(escalar_sku, 0) + 1
        self.assertEqual(len(escenario.lista_llenado), int(escenario.vector_inventario_inicial.sum()))
        escalar_total = len(escenario.lista_llenado)
        self.assertLessEqual(escalar_total, escalar_techo)
        for escalar_actividad, escalar_sku in escenario.lista_operacion:
            if escalar_actividad == 1:
                self.assertGreater(dic_contador.get(escalar_sku, 0), 0)
                dic_contador[escalar_sku] -= 1
                escalar_total -= 1
            else:
                dic_contador[escalar_sku] = dic_contador.get(escalar_sku, 0) + 1
                escalar_total += 1
                self.assertLessEqual(escalar_total, escalar_techo)

    def comprobar_retiros(self, escenario):
        # Cada retiro es una recuperación; tras el último retiro de un bloque la posición queda en cero
        # y la tarea siguiente de esa posición, si existe, es un almacenamiento (el primer lote del sucesor)
        lista = escenario.lista_operacion
        dic_contador = {}
        for _, escalar_sku in escenario.lista_llenado:
            dic_contador[escalar_sku] = dic_contador.get(escalar_sku, 0) + 1
        escalar_bloques = 0
        for escalar_indice, (escalar_actividad, escalar_sku) in enumerate(lista):
            dic_contador[escalar_sku] = dic_contador.get(escalar_sku, 0) + (1 if escalar_actividad == 0 else -1)
            if escalar_indice not in escenario.conjunto_retiros:
                continue
            self.assertEqual(escalar_actividad, 1)
            booleano_sigue_bloque = escalar_indice + 1 in escenario.conjunto_retiros and lista[escalar_indice + 1][1] == escalar_sku
            if booleano_sigue_bloque or escalar_indice == len(lista) - 1:
                continue # No es el último retiro del bloque, o el bloque quedó cortado al final de la lista
            escalar_bloques += 1
            self.assertEqual(dic_contador[escalar_sku], 0)
            for escalar_siguiente in range(escalar_indice + 1, len(lista)):
                if lista[escalar_siguiente][1] == escalar_sku:
                    self.assertEqual(lista[escalar_siguiente][0], 0)
                    break
        return escalar_bloques

    def test_kappa_y_lotes_del_caso_estatico(self):
        self.assertEqual(recuperaciones_diarias(cargar_configuracion()), 720)
        vector_demanda = 720 * demanda_hausman(180, 0.4)
        vector_lote = lotes_eoq(vector_demanda, calibrar_kappa(vector_demanda, 612))
        self.assertLessEqual(abs((vector_lote + 1).sum() / 2 - 612), 3)
        self.assertGreaterEqual(vector_lote.min(), 1)
        self.assertTrue(29 <= vector_lote[0] <= 33, vector_lote[0])
        self.assertTrue(4 <= np.median(vector_lote) <= 6, np.median(vector_lote))

    def test_largo_y_dias(self):
        escenario = self.escenario
        self.assertEqual(len(escenario.lista_operacion), 60 * TAREAS_POR_DIA)
        self.assertEqual(len(escenario.vector_dia_de_tarea), 60 * TAREAS_POR_DIA)
        self.assertEqual(escenario.vector_dia_de_tarea[0], 0)
        self.assertTrue(np.all(np.diff(escenario.vector_dia_de_tarea) >= 0))
        self.assertEqual(escenario.matriz_popularidad.shape[0], escenario.vector_dia_de_tarea[-1] + 1)
        self.assertEqual(escenario.resumen["dias_operacion"], escenario.matriz_popularidad.shape[0])

    def test_misma_semilla_mismo_escenario(self):
        escenario = generar(self.config)
        self.assertEqual(escenario.lista_llenado, self.escenario.lista_llenado)
        self.assertEqual(escenario.lista_operacion, self.escenario.lista_operacion)
        self.assertEqual(escenario.vector_dia_de_tarea, self.escenario.vector_dia_de_tarea)
        self.assertEqual(escenario.conjunto_retiros, self.escenario.conjunto_retiros)
        self.assertTrue(np.array_equal(escenario.matriz_popularidad, self.escenario.matriz_popularidad))
        self.assertTrue(resumenes_iguales(escenario.resumen, self.escenario.resumen))
        # Otra semilla da otro escenario
        self.assertNotEqual(generar(config_escenario(escalar_largo_secuencia=2000), 12).lista_operacion,
                            generar(config_escenario(escalar_largo_secuencia=2000), 11).lista_operacion)

    def test_misma_semilla_con_demanda_personalizada_y_estanteria_pequena(self):
        for config in (config_escenario(vector_configuracion_estanteria=np.array([6, 6, 2]), escalar_numero_skus=10, escalar_largo_secuencia=2000),
                       config_escenario(vector_configuracion_estanteria=np.array([6, 6, 2]), vector_demanda_sku_personalizada=[5, 3, 2],
                                        escalar_numero_skus=3, escalar_largo_secuencia=2000)):
            escenario_a, escenario_b = generar(config), generar(config)
            self.assertEqual(escenario_a.lista_llenado, escenario_b.lista_llenado)
            self.assertEqual(escenario_a.lista_operacion, escenario_b.lista_operacion)
            self.assertEqual(escenario_a.vector_dia_de_tarea, escenario_b.vector_dia_de_tarea)
            self.assertTrue(np.array_equal(escenario_a.matriz_popularidad, escenario_b.matriz_popularidad))
            self.assertTrue(resumenes_iguales(escenario_a.resumen, escenario_b.resumen))

    def test_inventario_nunca_negativo_ni_sobre_el_techo(self):
        self.comprobar_inventario(self.escenario, TECHO_INSTANCIA)

    def test_ocupacion_media_cerca_del_85(self):
        self.assertAlmostEqual(self.escenario.resumen["ocupacion_media"] / 720, 0.85, delta=0.03)
        self.assertLessEqual(self.escenario.resumen["ocupacion_maxima"], TECHO_INSTANCIA)

    def test_recuperaciones_de_demanda_por_dia(self):
        vector_completos = recuperaciones_por_dia(self.escenario)[:-1] # El último día puede quedar cortado
        self.assertGreaterEqual(len(vector_completos), 50)
        self.assertAlmostEqual(float(vector_completos.mean()), 720, delta=2)
        self.assertGreaterEqual(int(vector_completos.min()), 700)
        self.assertLessEqual(int(vector_completos.max()), 740)

    def test_mezcla_cerca_del_50(self):
        escalar_almacenes = sum(1 for escalar_actividad, _ in self.escenario.lista_operacion if escalar_actividad == 0)
        self.assertAlmostEqual(escalar_almacenes / len(self.escenario.lista_operacion), 0.5, delta=0.05)

    def test_demanda_no_atendida_excepcional(self):
        escalar_recuperaciones = int(recuperaciones_por_dia(self.escenario).sum())
        self.assertLessEqual(self.escenario.resumen["demanda_no_atendida"], 0.001 * escalar_recuperaciones)

    def test_popularidad_movil(self):
        # Popularidad del día 45: recuperaciones de demanda de los días 15 a 44, en las posiciones sin cierre en ese tramo
        cartera = crear_cartera(self.config, np.random.default_rng(11)) # La cartera es lo primero que sale del generador
        escalar_calentamiento = self.config["escalar_dias_calentamiento"]
        vector_sin_cierre = np.flatnonzero(cartera.generaciones(escalar_calentamiento + 15) == cartera.generaciones(escalar_calentamiento + 45))
        self.assertGreater(len(vector_sin_cierre), 100)
        for escalar_posicion in vector_sin_cierre[:20]:
            escalar_sku = int(escalar_posicion) + 1
            vector_cuenta = recuperaciones_por_dia(self.escenario, escalar_sku)
            self.assertEqual(int(self.escenario.matriz_popularidad[45, escalar_posicion]), int(vector_cuenta[15:45].sum()))
            self.assertTrue(np.array_equal(self.escenario.popularidad(45), self.escenario.matriz_popularidad[45]))
        # El día se acota a la última fila generada
        self.assertTrue(np.array_equal(self.escenario.popularidad(10_000), self.escenario.matriz_popularidad[-1]))

    def test_retiros_vacian_la_posicion(self):
        self.assertGreater(self.escenario.resumen["lanzamientos"], 0)
        self.assertGreater(self.comprobar_retiros(self.escenario), 0)
        self.assertEqual(self.escenario.resumen["retiros"], len(self.escenario.conjunto_retiros))

    def test_modelo_estatico_sin_retiros_ni_lanzamientos(self):
        escenario = generar(config_escenario(texto_modelo_demanda="estatica", escalar_largo_secuencia=20 * TAREAS_POR_DIA))
        self.assertEqual(escenario.conjunto_retiros, set())
        self.assertEqual(escenario.resumen["lanzamientos"], 0)
        self.assertEqual(escenario.resumen["retiros"], 0)
        self.comprobar_inventario(escenario, TECHO_INSTANCIA)

    def test_lista_corta(self):
        # Foco 1: una lista más corta que un día
        escenario = generar(config_escenario(escalar_largo_secuencia=72))
        self.assertEqual(len(escenario.lista_operacion), 72)
        self.assertEqual(escenario.vector_dia_de_tarea, [0] * 72)
        self.assertEqual(escenario.matriz_popularidad.shape[0], 1)
        self.assertEqual(escenario.resumen["dias_operacion"], 1)
        self.comprobar_inventario(escenario, TECHO_INSTANCIA)

    def test_varias_sucesiones(self):
        # Foco 2: con ciclos de un mes cada posición cierra varias veces en 90 días
        config = config_escenario(escalar_meses_ciclo_vida=1, escalar_largo_secuencia=90 * TAREAS_POR_DIA)
        escenario = generar(config)
        self.assertGreaterEqual(escenario.resumen["lanzamientos"], 180 * 2)
        self.assertGreaterEqual(self.comprobar_retiros(escenario), 180)
        self.comprobar_inventario(escenario, TECHO_INSTANCIA)
        # La popularidad del sucesor arranca en cero el día del lanzamiento y al día siguiente cuenta solo ese día
        cartera = crear_cartera(config, np.random.default_rng(11))
        escalar_calentamiento = config["escalar_dias_calentamiento"]
        escalar_dias = escenario.matriz_popularidad.shape[0]
        escalar_revisados = 0
        for escalar_posicion in range(0, 180, 9):
            vector_cuenta = recuperaciones_por_dia(escenario, escalar_posicion + 1)
            for escalar_dia in range(1, escalar_dias - 2):
                escalar_generacion = cartera.generaciones(escalar_calentamiento + escalar_dia)[escalar_posicion]
                if escalar_generacion > cartera.generaciones(escalar_calentamiento + escalar_dia - 1)[escalar_posicion]:
                    self.assertEqual(int(escenario.matriz_popularidad[escalar_dia, escalar_posicion]), 0)
                    self.assertEqual(int(escenario.matriz_popularidad[escalar_dia + 1, escalar_posicion]), int(vector_cuenta[escalar_dia]))
                    escalar_revisados += 1
        self.assertGreater(escalar_revisados, 20)

    def test_techo_activo(self):
        # Foco 3: con una ocupación media cercana al techo los lotes se aplazan, pero el generador termina y no pasa el techo
        escenario = generar(config_escenario(escalar_nivel_ocupacion=0.93))
        self.assertEqual(len(escenario.lista_operacion), 60 * TAREAS_POR_DIA)
        self.comprobar_inventario(escenario, TECHO_INSTANCIA)
        self.assertGreater(escenario.resumen["lotes_aplazados"], 0)
        self.assertLessEqual(escenario.resumen["ocupacion_maxima"], TECHO_INSTANCIA)

    def test_estanteria_pequena_y_demanda_personalizada(self):
        # Foco 4: la estantería pequeña de la herramienta web, con la demanda de Hausman y con una demanda personalizada
        for config in (config_escenario(vector_configuracion_estanteria=np.array([6, 6, 2]), escalar_numero_skus=10, escalar_largo_secuencia=2000),
                       config_escenario(vector_configuracion_estanteria=np.array([6, 6, 2]), vector_demanda_sku_personalizada=[5, 3, 2],
                                        escalar_numero_skus=3, escalar_largo_secuencia=2000)):
            escenario = generar(config)
            self.assertEqual(len(escenario.lista_operacion), 2000)
            self.comprobar_inventario(escenario, 68) # floor(0,95 · 72)
            # Los lotes de cada día son de al menos una unidad
            cartera = crear_cartera(config, np.random.default_rng(11))
            escalar_n = len(cartera.vector_altura)
            self.assertEqual(len(escenario.vector_inventario_inicial), escalar_n)
            for escalar_dia in range(config["escalar_dias_calentamiento"] + escenario.resumen["dias_operacion"]):
                vector_demanda = recuperaciones_diarias(config) * cartera.participaciones(escalar_dia)
                vector_lote = lotes_eoq(vector_demanda, calibrar_kappa(vector_demanda, 0.85 * 72))
                self.assertGreaterEqual(int(vector_lote.min()), 1)
            # Cada SKU recibe almacenamientos en la operación
            self.assertEqual({escalar_sku for escalar_actividad, escalar_sku in escenario.lista_operacion if escalar_actividad == 0},
                             set(range(1, escalar_n + 1)))


class EscenarioInvertido:
    # Escenario mínimo: antes de escalar_dia_cambio la posición 1 es la más popular y desde ese día lo es la última
    def __init__(self, escalar_skus=180, escalar_dia_cambio=30):
        self.escalar_skus = escalar_skus
        self.escalar_dia_cambio = escalar_dia_cambio

    def popularidad(self, escalar_dia):
        vector_pop = np.arange(self.escalar_skus, 0, -1, dtype=int) # n, n - 1, ..., 1
        return vector_pop[::-1].copy() if escalar_dia >= self.escalar_dia_cambio else vector_pop


class PruebasPoliticasConPopularidad(unittest.TestCase):
    CLAVES_CON_ZONAS = ('basada_en_clases', 'full_turnover')
    CLAVES_REPLANIFICADAS = ('basada_en_clases_replanificada', 'full_turnover_replanificada')

    def crear(self, txt_clave, escenario="por_defecto", **dic_cambios):
        # Política sobre una estantería vacía, con el escenario mínimo (o ninguno) en la configuración
        config = cargar_configuracion()
        config.update(dic_cambios)
        if escenario == "por_defecto":
            escenario = EscenarioInvertido(config["escalar_numero_skus"])
        if escenario is not None:
            config["escenario_demanda"] = escenario
        E, G, P = Simulacion(config).crearInstancias()
        return crear_politica(txt_clave, E, G, P, config)

    def etiqueta_mas_cercana(self, politica):
        # Etiqueta de la celda más cercana al E/S, la primera que reparte la zonificación (menor tiempo, fila, columna y capa)
        return int(politica.matriz_zonificacion[np.unravel_index(np.argmin(politica.tiempoChebyshevXY(politica.instancia_puntoES.vector_posicion)),
                                                                 politica.matriz_zonificacion.shape)])

    def plan(self, politica):
        # Lo que la política planificó: la zonificación y, en la basada en clases, la clase de cada SKU
        vector_clases = getattr(politica, "vector_clase_de_sku", np.zeros(0))
        return politica.matriz_zonificacion.tobytes(), vector_clases.tobytes()

    def test_variantes_registradas(self):
        self.assertEqual(set(POLITICAS_ASIGNACION), {'aleatoria', 'basada_en_clases', 'basada_en_clases_replanificada', 'full_turnover',
                                                     'full_turnover_replanificada', 'mas_cercana_es', 'mas_cercana_grua'})
        self.assertEqual(tuple(cargar_configuracion()["vector_politicas_asignacion"]), tuple(POLITICAS_ASIGNACION))
        for txt_clave in POLITICAS_ASIGNACION:
            for escenario in ("por_defecto", None):
                self.assertTrue(self.crear(txt_clave, escenario).nombre, (txt_clave, escenario))
        self.assertEqual({self.crear(txt_clave).nombre for txt_clave in self.CLAVES_REPLANIFICADAS},
                         {'BasadaEnClasesReplanificada', 'FullTurnoverReplanificada'})

    def test_plan_fijo_no_cambia_sus_zonas(self):
        for txt_clave in self.CLAVES_CON_ZONAS:
            politica = self.crear(txt_clave)
            matriz_inicial = politica.matriz_zonificacion.copy()
            vector_clases = getattr(politica, "vector_clase_de_sku", None)
            vector_clases = None if vector_clases is None else vector_clases.copy()
            for escalar_dia in (0, 30, 60):
                politica.iniciarDia(escalar_dia)
                self.assertTrue(np.array_equal(politica.matriz_zonificacion, matriz_inicial), (txt_clave, escalar_dia))
                if vector_clases is not None:
                    self.assertTrue(np.array_equal(politica.vector_clase_de_sku, vector_clases), (txt_clave, escalar_dia))

    def test_replanificada_cambia_sus_zonas_solo_cada_30_dias(self):
        escalar_n = cargar_configuracion()["escalar_numero_skus"]
        for txt_clave in self.CLAVES_REPLANIFICADAS:
            politica = self.crear(txt_clave)
            matriz_inicial = politica.matriz_zonificacion.copy()
            # Plan inicial: la posición 1 es la más popular y tiene la zona más cercana al E/S
            self.assertEqual(self.etiqueta_mas_cercana(politica), politica.etiquetaZona(1), txt_clave)
            for escalar_dia in (0, 15, 29):
                politica.iniciarDia(escalar_dia)
                self.assertTrue(np.array_equal(politica.matriz_zonificacion, matriz_inicial), (txt_clave, escalar_dia))
            politica.iniciarDia(30)
            # Ahora la última posición es la más popular y su zona queda en la celda más cercana
            self.assertEqual(self.etiqueta_mas_cercana(politica), politica.etiquetaZona(escalar_n), txt_clave)
            if "clases" in txt_clave:
                self.assertEqual((politica.etiquetaZona(escalar_n), politica.etiquetaZona(1)), (1, 3))
                self.assertEqual(self.etiqueta_mas_cercana(politica), 1)
            else:
                self.assertEqual(self.etiqueta_mas_cercana(politica), escalar_n)
                self.assertFalse(np.array_equal(politica.matriz_zonificacion, matriz_inicial))

    def test_la_primera_replanificacion_ocurre_en_el_periodo_configurado(self):
        escalar_n = cargar_configuracion()["escalar_numero_skus"]
        for txt_clave in self.CLAVES_REPLANIFICADAS:
            # Con un periodo de 10 días la popularidad se invierte en el día 10 y el plan cambia ahí, no en el 30
            politica = self.crear(txt_clave, EscenarioInvertido(escalar_n, 10), escalar_dias_replanificacion=10)
            tupla_antes = self.plan(politica)
            for escalar_dia in (0, 5, 9):
                politica.iniciarDia(escalar_dia)
                self.assertEqual(self.plan(politica), tupla_antes, (txt_clave, escalar_dia))
            politica.iniciarDia(10)
            self.assertNotEqual(self.plan(politica), tupla_antes, txt_clave)
            self.assertEqual(self.etiqueta_mas_cercana(politica), politica.etiquetaZona(escalar_n), txt_clave)

    def test_un_dia_omitido_no_salta_la_replanificacion(self):
        escalar_n = cargar_configuracion()["escalar_numero_skus"]
        for txt_clave in self.CLAVES_REPLANIFICADAS:
            politica = self.crear(txt_clave)
            tupla_inicial = self.plan(politica)
            politica.iniciarDia(29)
            self.assertEqual(self.plan(politica), tupla_inicial, txt_clave)
            # El día 30 no llega al controlador: el 31 ya pertenece al periodo siguiente y replanifica
            politica.iniciarDia(31)
            tupla_nueva = self.plan(politica)
            self.assertNotEqual(tupla_nueva, tupla_inicial, txt_clave)
            self.assertEqual(self.etiqueta_mas_cercana(politica), politica.etiquetaZona(escalar_n), txt_clave)
            # Dentro del mismo periodo no vuelve a replanificar
            politica.iniciarDia(32)
            self.assertEqual(self.plan(politica), tupla_nueva, txt_clave)

    def test_replanificada_sin_escenario_no_hace_nada(self):
        for txt_clave in self.CLAVES_REPLANIFICADAS:
            politica = self.crear(txt_clave, None)
            matriz_inicial = politica.matriz_zonificacion.copy()
            for escalar_dia in (0, 30, 60):
                politica.iniciarDia(escalar_dia)
            self.assertTrue(np.array_equal(politica.matriz_zonificacion, matriz_inicial), txt_clave)

    def test_planifica_con_la_popularidad_del_primer_dia(self):
        escenario = EscenarioInvertido(180)
        for txt_clave in ('basada_en_clases', 'basada_en_clases_replanificada'):
            politica = self.crear(txt_clave, escenario)
            self.assertTrue(np.array_equal(politica.vector_clase_de_sku, clasificar_skus(escenario.popularidad(0), politica.config)), txt_clave)
        # Sin escenario planifican con la demanda de la configuración, como antes
        politica = self.crear('basada_en_clases', None)
        self.assertTrue(np.array_equal(politica.vector_clase_de_sku, clasificar_skus(politica.config["vector_demanda_sku"], politica.config)))

    def test_popularidad_degenerada_cae_a_la_demanda_de_la_configuracion(self):
        # Sin recuperaciones en la ventana la popularidad es cero en todas las posiciones: no se puede planificar con ella
        class EscenarioSinDemanda:
            def popularidad(self, escalar_dia):
                return np.zeros(180, dtype=int)

        for txt_criterio in ("skus", "iguales", "pareto"):
            for txt_clave in POLITICAS_ASIGNACION:
                politica = self.crear(txt_clave, EscenarioSinDemanda(), texto_criterio_clases=txt_criterio)
                if txt_clave.startswith('basada_en_clases'):
                    self.assertTrue(np.array_equal(politica.vector_clase_de_sku, clasificar_skus(politica.config["vector_demanda_sku"], politica.config)),
                                    (txt_clave, txt_criterio))
                politica.iniciarDia(30)
            vector_demanda = politica.config["vector_demanda_sku"]
            self.assertTrue(np.array_equal(politica.demandaPlanificacion(0), vector_demanda))
            self.assertTrue(np.all(np.isfinite(politica.demandaPlanificacion(30))))


class PruebasReplanificacionDeExtremoAExtremo(unittest.TestCase):
    def test_un_escenario_generado_replanifica_en_el_dia_30_por_el_controlador(self):
        import random
        from asrs.control.controlador_grua import ControladorGrua
        from asrs.politicas.basada_en_clases import PoliticaBasadaEnClasesReplanificada

        class Espia(PoliticaBasadaEnClasesReplanificada):
            # Anota el día de cada llamada a planificar, None es la del plan inicial
            ultimo_dia = None

            def planificar(self, vector_demanda):
                self.__dict__.setdefault("vector_dias_planificados", []).append(self.ultimo_dia)
                super().planificar(vector_demanda)

            def iniciarDia(self, escalar_dia):
                self.ultimo_dia = escalar_dia
                super().iniciarDia(escalar_dia)

        # Pocos ciclos por hora: unas 48 tareas por día, así 1 700 tareas cubren el día 30 pero no el 60
        config = cargar_configuracion()
        config.update(escalar_ciclos_por_hora=2, escalar_largo_secuencia=1_700)
        simulacion = Simulacion(config)
        random.seed(5)
        SA, S = simulacion.generarSecuencias()
        vector_dias = simulacion.escenario_demanda.vector_dia_de_tarea
        self.assertTrue(30 <= max(vector_dias) < 60, max(vector_dias))
        E, G, P = simulacion.crearInstancias()
        politica = Espia(E, G, P, simulacion.configPolitica())
        ControladorGrua(E, G, P, SA, politica).ejecutarSecuencia() # El llenado no avisa días
        ControladorGrua(E, G, P, S, politica, **simulacion.argumentosOperacion(len(S))).ejecutarSecuencia()
        self.assertEqual(politica.vector_dias_planificados, [None, 30])


if __name__ == "__main__":
    unittest.main()
