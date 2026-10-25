"""Pruebas de regresión del simulador ASRS. Ejecutar con: python -m unittest discover -s tests"""
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asrs.config import cargar_configuracion as _cargar_configuracion_instancia
from asrs.dominio.grua import Grua
from asrs.dominio.item import Item
from asrs.politicas.recuperacion import PoliticaRecuperacionFCFS, PoliticaRecuperacionFCFSAccesible
from asrs.politicas.registro import POLITICAS_ASIGNACION, crear_politica
from asrs.politicas.restriccion import MatricesRestriccion
from asrs.politicas.zonificacion import agrupar_skus_en_clases, construir_zonificacion
from asrs.simulacion.simulacion import Simulacion as _Simulacion

# Las pruebas usan los parámetros de la instancia del capítulo 6, pero con una lista de operación corta (el 10 % de las celdas)
# para que se ejecuten en segundos: la instancia completa evalúa 100 000 tareas por corrida
TAREAS_PRUEBA = 72


def cargar_configuracion():
    config = _cargar_configuracion_instancia()
    config["escalar_largo_secuencia"] = TAREAS_PRUEBA
    return config


class Simulacion(_Simulacion):
    def __init__(self, config=None):
        super().__init__(config if config is not None else cargar_configuracion())


def preparar_estado(txt_clave_politica):
    # Estantería llena al nivel configurado bajo la política dada
    simulacion = Simulacion()
    SA, S = simulacion.generarSecuencias()
    random.seed(1)
    from asrs.control.controlador_grua import ControladorGrua
    Item.escalar_contador_global_generacion = 0
    E, G, P = simulacion.crearInstancias()
    politica = crear_politica(txt_clave_politica, E, G, P, simulacion.config)
    ControladorGrua(E, G, P, SA, politica).ejecutarSecuencia()
    return simulacion, E, G, P, politica


class PruebasCinematica(unittest.TestCase):
    def test_sin_aceleracion_es_velocidad_constante(self):
        from asrs.cinematica import tiempo_recorrido
        self.assertAlmostEqual(float(tiempo_recorrido(10, 2)), 5.0)
        self.assertAlmostEqual(float(tiempo_recorrido(10, 2, 0, 0)), 5.0)

    def test_perfil_trapezoidal(self):
        from asrs.cinematica import tiempo_recorrido
        # v=2, a=1, d=1: 2 s acelerando (2 m), 2 s frenando (2 m) y 6 m a velocidad constante (3 s)
        self.assertAlmostEqual(float(tiempo_recorrido(10, 2, 1, 1)), 7.0)

    def test_perfil_triangular_si_no_alcanza_la_velocidad(self):
        from asrs.cinematica import tiempo_recorrido
        self.assertAlmostEqual(float(tiempo_recorrido(2, 2, 1, 1)), 2 * np.sqrt(2.0))

    def test_aceleracion_y_deceleracion_distintas(self):
        from asrs.cinematica import tiempo_recorrido
        # v=2, a=1 (2 s, 2 m), d=0.5 (4 s, 4 m), 4 m a velocidad constante (2 s)
        self.assertAlmostEqual(float(tiempo_recorrido(10, 2, 1, 0.5)), 8.0)
        # solo hay límite al acelerar, frenar es instantáneo
        self.assertAlmostEqual(float(tiempo_recorrido(10, 2, 1, 0)), 6.0)

    def test_acelerar_nunca_acorta_el_tiempo_y_la_distancia_cero_es_gratis(self):
        from asrs.cinematica import tiempo_recorrido
        distancias = np.linspace(0, 30, 31)
        self.assertTrue(np.all(tiempo_recorrido(distancias, 1.5, 1, 1) >= tiempo_recorrido(distancias, 1.5) - 1e-12))
        self.assertEqual(float(tiempo_recorrido(0, 1.5, 1, 1)), 0.0)

    def test_version_escalar_coincide_con_la_de_arreglos(self):
        from asrs.cinematica import tiempo_recorrido, tiempo_recorrido_escalar
        generador = np.random.default_rng(0)
        for _ in range(500):
            distancia, velocidad = generador.uniform(0, 30), generador.uniform(0.2, 3)
            aceleracion, deceleracion = [generador.choice([0.0, generador.uniform(0.1, 5)]) for _ in range(2)]
            self.assertAlmostEqual(tiempo_recorrido_escalar(distancia, velocidad, aceleracion, deceleracion),
                                   float(tiempo_recorrido(distancia, velocidad, aceleracion, deceleracion)), places=9)

    def test_tiempo_chebyshev_toma_el_eje_mas_lento(self):
        from asrs.cinematica import tiempo_chebyshev_xy
        self.assertAlmostEqual(float(tiempo_chebyshev_xy(np.array([2.0, 10.0]), np.array([2.0, 2.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))), 7.0)


class PruebasGrua(unittest.TestCase):
    def setUp(self):
        self.E, self.G, self.P = Simulacion().crearInstancias()

    def test_volver_a_plano_cuenta_tiempo(self):
        grua = Grua(np.array([0, 0, 2.7]), np.array([1, 1.5, 0.7]), 0)
        grua.entrarAlPasillo(self.E)
        grua.moverEnEstanteria((3, 3, 0))
        self.assertGreater(grua.volverAplano(), 0)

    def test_la_aceleracion_alarga_los_desplazamientos(self):
        sin = Grua(np.array([0, 0, 2.7]), np.array([1, 1.5, 0.7]), 0)
        con = Grua(np.array([0, 0, 2.7]), np.array([1, 1.5, 0.7]), 0, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        origen, destino = np.array([0.0, 0.0, 2.7]), np.array([9.6, 12.6, 0.0])
        self.assertAlmostEqual(sin.intancia_metricas.tiempoDezplamiento(origen, destino, sin.vector_velocidad), max(9.6 / 1.0, 12.6 / 1.5) + 2.7 / 0.7)
        self.assertGreater(con._tiempo(origen, destino), sin._tiempo(origen, destino))

    def test_tiempo_de_toma_y_deposito_depende_de_la_capa(self):
        # Instancia del capítulo 6: 3,43 s en el E/S y en la capa delantera, 6,86 s en la trasera
        config = cargar_configuracion()
        simulacion = Simulacion(config)
        E, G, P = simulacion.crearInstancias()
        G.entrarAlPasillo(E)
        self.assertAlmostEqual(G.tiempo_capa_delantera, 3.43)
        E.matriz_contenedor_items[3, 4, 1] = Item(1)   # capa delantera (mayor índice)
        E.matriz_contenedor_items[3, 5, 0] = Item(2)   # capa trasera
        self.assertAlmostEqual(G.cargarItemDeEstanteria((3, 4, 1)), 3.43)
        self.assertAlmostEqual(G.cargarItemDeEstanteria((3, 5, 0)), 6.86)
        self.assertAlmostEqual(G.soltarItemEnEstanteria((3, 5, 0)), 6.86)
        self.assertAlmostEqual(G.soltarItemEnEstanteria((3, 4, 1)), 3.43)
        P.matriz_contenedor_temporal_items.append(Item(3))
        G.irAPuntoES(P)
        self.assertAlmostEqual(G.cargarItemPuntoES(), 3.43)

    def test_la_instancia_por_defecto_es_la_del_capitulo_6(self):
        config = _cargar_configuracion_instancia()
        self.assertEqual(config["vector_configuracion_estanteria"].tolist(), [18, 20, 2])
        self.assertEqual(config["vector_velocidad_grua"].tolist(), [1, 1.5, 0.7])
        self.assertEqual(config["vector_aceleracion_grua"].tolist(), [0.5, 0.5, 0])
        self.assertEqual(config["escalar_numero_skus"], 180)
        self.assertEqual(config["escalar_sigma_hausman"], 0.4)
        self.assertEqual(config["escalar_largo_secuencia"], 100_000)
        self.assertEqual(config["vector_proporcion_skus_por_clase"].tolist(), [0.2, 0.2, 0.6])
        # El E/S queda 1 m sobre la base de la estantería y 1 m antes de la primera columna, con el riel en la cara frontal
        E, G, P = Simulacion(_cargar_configuracion_instancia()).crearInstancias()
        self.assertEqual(E.vector_posicion.tolist(), [-1.0, 1.0, 0])
        self.assertAlmostEqual(G.vector_posicion_resposo[2], 2 * 1.2)

    def test_posicion_riel_no_se_trunca(self):
        grua = Grua(np.array([0, 0, 2.7]), np.array([1, 1.5, 0.7]), 0)
        grua.entrarAlPasillo(self.E)
        self.assertAlmostEqual(grua.vector_posicion_actual[2], 2.7)

    def test_estanteria_desplazada_riel_delante_y_sin_pasar_por_la_esquina(self):
        from asrs.control.controlador_grua import ControladorGrua
        config = cargar_configuracion()
        config["vector_posicion_estanteria"] = np.array([0.0, 5.0, 3.0])
        simulacion = Simulacion(config)
        E, G, P = simulacion.crearInstancias()
        # El riel queda delante de la cara frontal de la estantería, también cuando esta se desplaza en profundidad
        self.assertAlmostEqual(G.vector_posicion_resposo[2], 3.0 + 2 * config["vector_cotas_estanteria"][2] + config["escalar_separacion_riel_grua"])
        self.assertTrue((E.posicionesCeldas()[:, :, :, 2] < G.vector_posicion_resposo[2]).all())
        # Al almacenar, la grúa va del E/S a la celda sin detenerse en la esquina de la estantería (0, 5)
        politica = crear_politica('mas_cercana_es', E, G, P, config)
        G.vector_eventos = []
        P.matriz_contenedor_temporal_items.append(Item(1))
        ControladorGrua(E, G, P, [], politica).almacenar((4, 6, 0))
        paradas = {(e[5], e[6]) for e in G.vector_eventos if e[1] == 'M'}
        self.assertNotIn((0.0, 5.0), paradas)


class PruebasRestriccion(unittest.TestCase):
    def test_accesibilidad(self):
        ocupacion = np.zeros((1, 1, 3), dtype=bool)
        ocupacion[0, 0, 2] = True  # ocupada la capa frontal
        accesible = MatricesRestriccion().accesibilidad(ocupacion)
        self.assertEqual(accesible[0, 0].tolist(), [False, False, True])

    def test_ocupacion(self):
        simulacion = Simulacion()
        E, _, _ = simulacion.crearInstancias()
        E.matriz_contenedor_items[0, 0, 0] = Item(1)
        ocupacion = MatricesRestriccion().ocupacion(E)
        self.assertEqual(int(ocupacion.sum()), 1)
        self.assertTrue(ocupacion[0, 0, 0])


class PruebasZonificacion(unittest.TestCase):
    def test_zonas_por_canales_completos_y_en_orden(self):
        # 4 filas x 2 columnas x 2 capas, todas las capas de un canal tienen el mismo tiempo
        tiempos = np.repeat(np.arange(8, dtype=float).reshape(4, 2, 1), 2, axis=2)
        zonas = construir_zonificacion(tiempos, [0.25, 0.75], [2, 1])
        # cada canal pertenece por completo a una etiqueta
        self.assertTrue(np.all(zonas[:, :, 0] == zonas[:, :, 1]))
        # cada etiqueta recibe al menos un canal y todas las celdas quedan asignadas
        self.assertGreaterEqual(int((zonas == 1).sum()), 2)
        self.assertEqual(int((zonas == 1).sum() + (zonas == 2).sum()), tiempos.size)
        # la etiqueta 2 va primera en el orden, así que recibe las celdas de menor tiempo
        self.assertTrue(np.all(tiempos[zonas == 2] < tiempos[zonas == 1].min()))
        self.assertGreater(int((zonas == 2).sum()), int((zonas == 1).sum()))

    def test_cada_etiqueta_recibe_al_menos_un_canal(self):
        tiempos = np.repeat(np.arange(6, dtype=float).reshape(3, 2, 1), 2, axis=2)
        zonas = construir_zonificacion(tiempos, [0.98, 0.01, 0.01], [1, 2, 3])
        for etiqueta in (1, 2, 3):
            self.assertGreaterEqual(int((zonas == etiqueta).sum()), 2)

    def test_agrupar_skus_en_clases_por_demanda(self):
        demanda = np.array([0.05, 0.4, 0.1, 0.3, 0.15])           # el SKU 2 es el más demandado, luego el 4
        clases = agrupar_skus_en_clases(demanda, [0.4, 0.6])        # 2 SKUs en la clase 1 y 3 en la clase 2
        self.assertEqual(clases.tolist(), [2, 1, 2, 1, 2])
        # cada clase recibe al menos un SKU aunque su proporción sea mínima
        self.assertEqual(sorted(set(agrupar_skus_en_clases(np.ones(10) / 10, [0.99, 0.005, 0.005]).tolist())), [1, 2, 3])


class PruebasGeneradorTareas(unittest.TestCase):
    def test_las_tareas_son_de_skus_con_cantidades_exactas(self):
        from asrs.generacion.secuencia import GeneradorSecuencia
        demanda = np.array([0.5, 0.3, 0.2])
        random.seed(0)
        tareas = GeneradorSecuencia(100, np.array([0.6, 1]), np.cumsum(demanda)).generarSecuencia()
        self.assertEqual(len(tareas), 100)
        self.assertEqual(sum(1 for actividad, _ in tareas if actividad == 0), 60)
        self.assertEqual(sorted({sku for _, sku in tareas}), [1, 2, 3])
        almacenar = [sku for actividad, sku in tareas if actividad == 0]
        self.assertEqual([almacenar.count(sku) for sku in (1, 2, 3)], [30, 18, 12])

    def test_misma_semilla_misma_lista(self):
        from asrs.generacion.secuencia import GeneradorSecuencia
        acumulada = np.cumsum([0.25, 0.25, 0.5])
        random.seed(5); a = GeneradorSecuencia(40, np.array([0.5, 1]), acumulada).generarSecuencia()
        random.seed(5); b = GeneradorSecuencia(40, np.array([0.5, 1]), acumulada).generarSecuencia()
        self.assertEqual(a, b)

    def test_demanda_de_hausman(self):
        from asrs.generacion.probabilidades import demanda_hausman
        uniforme = demanda_hausman(10, 1.0)
        self.assertTrue(np.allclose(uniforme, 0.1))
        sesgada = demanda_hausman(10, 0.5)
        self.assertAlmostEqual(float(sesgada.sum()), 1.0)
        self.assertTrue(np.all(np.diff(sesgada) < 0))          # el SKU 1 es el más demandado

    def test_perfil_de_inventario(self):
        from asrs.generacion.probabilidades import perfil_inventario
        demanda = np.array([0.64, 0.16, 0.16, 0.04])
        # EOQ: el inventario crece con la raíz de la demanda (0.8, 0.4, 0.4, 0.2 normalizado)
        self.assertTrue(np.allclose(perfil_inventario(demanda, "eoq"), np.array([0.8, 0.4, 0.4, 0.2]) / 1.8))
        self.assertTrue(np.allclose(perfil_inventario(demanda, "demanda"), demanda))
        # con EOQ el SKU más demandado rota más rápido por unidad guardada; proporcional, todos rotan igual
        self.assertGreater(*(demanda / perfil_inventario(demanda, "eoq"))[[0, 3]])
        self.assertTrue(np.allclose(demanda / perfil_inventario(demanda, "demanda"), 1))

    def test_el_llenado_sigue_el_perfil_de_inventario(self):
        from asrs.generacion.probabilidades import perfil_inventario
        from asrs.utilidades import repartir_entero
        for txt_perfil in ("eoq", "demanda"):
            simulacion = Simulacion(dict(cargar_configuracion(), texto_perfil_inventario=txt_perfil))
            random.seed(3)
            SA, S = simulacion.generarSecuencias()
            cfg = simulacion.config
            conteo = [sum(1 for _, sku in SA if sku == i) for i in range(1, cfg["escalar_numero_skus"] + 1)]
            esperado = repartir_entero(len(SA), perfil_inventario(cfg["vector_demanda_sku"], txt_perfil))
            self.assertEqual(conteo, list(esperado), txt_perfil)


class PruebasDatosReales(unittest.TestCase):
    CSV = "SKU;Task\nX1;A\nX2;A\nX1;R\nX1;A\nX2;R\nZ9;A\nX1;R\n"

    def test_lee_encabezado_separador_y_actividades(self):
        from asrs.generacion.datos_reales import leer_tareas
        tareas = leer_tareas(self.CSV)
        self.assertEqual(tareas[:3], [(0, "X1"), (0, "X2"), (1, "X1")])
        self.assertEqual(len(tareas), 7)
        # sin encabezado, con coma y palabras completas
        self.assertEqual(leer_tareas("P1,entrada\nP1,salida"), [(0, "P1"), (1, "P1")])
        # SKU numéricos: se toma el orden SKU y Task aunque "1" también sea una actividad válida
        self.assertEqual(leer_tareas("1;R\n2;A"), [(1, "1"), (0, "2")])
        # encabezado con las columnas en otro orden, o sin encabezado y con la actividad primero
        self.assertEqual(leer_tareas("Task,SKU\nR,P1"), [(1, "P1")])
        self.assertEqual(leer_tareas("A;P1\nR;P1"), [(0, "P1"), (1, "P1")])

    def test_errores_claros(self):
        from asrs.generacion.datos_reales import leer_tareas
        for texto in ("", "solo una columna", "SKU;Task\nX1;Q", "SKU;Task\n;A", "x;y;z\n1;2;3"):
            with self.assertRaises(ValueError, msg=texto):
                leer_tareas(texto)

    def test_numera_los_skus_por_frecuencia_en_el_historial(self):
        from asrs.generacion.datos_reales import leer_tareas, preparar_datos_reales
        secuencia, demanda, etiquetas = preparar_datos_reales(leer_tareas(self.CSV), 100)
        self.assertEqual(etiquetas, ["X1", "X2", "Z9"])                  # X1 aparece 4 veces, X2 2 y Z9 1
        self.assertAlmostEqual(float(demanda.sum()), 1.0)
        self.assertEqual(secuencia[0], (0, 1))
        # con historial corto, Z9 (que solo aparece al final) no tiene demanda conocida y queda al final
        secuencia, demanda, etiquetas = preparar_datos_reales(leer_tareas(self.CSV), 50)
        self.assertEqual(etiquetas[-1], "Z9")
        self.assertEqual(float(demanda[-1]), 0.0)

    def test_la_simulacion_usa_la_lista_real(self):
        from asrs.config import config_a_json, config_desde_json
        datos = config_a_json(cargar_configuracion())
        datos.update(texto_origen_tareas="datos", texto_tareas_csv=self.CSV, escalar_porcentaje_historial=50)
        cfg = config_desde_json(datos)
        self.assertEqual(cfg["escalar_largo_secuencia"], 7)
        self.assertEqual(cfg["escalar_numero_skus"], 3)
        random.seed(1)
        _, tareas = Simulacion(cfg).generarSecuencias()
        self.assertEqual(tareas, [tuple(t) for t in cfg["secuencia_real"]])
        resultado = Simulacion.cargarReplica((cfg, 1))
        self.assertEqual(len(resultado), len(cfg["vector_politicas_asignacion"]) * len(cfg["vector_politicas_recuperacion"]))

    def test_datos_invalidos_se_rechazan_en_la_configuracion(self):
        from asrs.config import config_a_json, config_desde_json
        datos = config_a_json(cargar_configuracion())
        with self.assertRaises(ValueError):
            config_desde_json({**datos, "texto_origen_tareas": "datos", "texto_tareas_csv": ""})
        with self.assertRaises(ValueError):
            config_desde_json({**datos, "texto_origen_tareas": "otro"})


class PruebasPoliticas(unittest.TestCase):
    def test_todas_las_politicas_registradas_se_construyen(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        for clave in POLITICAS_ASIGNACION:
            self.assertTrue(crear_politica(clave, E, G, P, simulacion.config).nombre)

    def test_minimizacion_de_varianza_llena_desde_el_fondo(self):
        # En una estantería vacía, la celda elegida por cualquier política es de la capa más profunda
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        for clave in POLITICAS_ASIGNACION:
            politica = crear_politica(clave, E, G, P, simulacion.config)
            self.assertEqual(politica.asignar(1)[2], 0, clave)

    def test_mas_cercana_al_es_elige_el_canal_mas_cercano(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        politica = crear_politica('mas_cercana_es', E, G, P, simulacion.config)
        self.assertEqual(politica.asignar(1), (0, 0, 0))

    def test_politica_desconocida(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        with self.assertRaises(ValueError):
            crear_politica('inexistente', E, G, P, simulacion.config)

    def test_full_turnover_da_una_zona_a_cada_sku_ordenada_por_demanda(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        config = dict(simulacion.config, vector_demanda_sku=np.array([0.1, 0.2, 0.7]), escalar_numero_skus=3)
        politica = crear_politica('full_turnover', E, G, P, config)
        self.assertEqual(politica.matriz_zonificacion[0, 0, 0], 3)            # el SKU más demandado queda en la celda más cercana
        self.assertEqual(sorted(set(politica.matriz_zonificacion.ravel().tolist())), [1, 2, 3])
        self.assertEqual(politica.etiquetaZona(2), 2)

    def test_las_zonas_se_dimensionan_con_el_inventario(self):
        from asrs.generacion.probabilidades import perfil_inventario
        from asrs.utilidades import repartir_entero
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        demanda = np.array([0.64, 0.16, 0.16, 0.04])
        canales = E.matriz_contenedor_items.shape[0] * E.matriz_contenedor_items.shape[1]
        for txt_perfil in ("eoq", "demanda"):
            config = dict(simulacion.config, vector_demanda_sku=demanda, escalar_numero_skus=4, texto_perfil_inventario=txt_perfil)
            politica = crear_politica('full_turnover', E, G, P, config)
            canales_por_sku = [int((politica.matriz_zonificacion[:, :, 0] == sku).sum()) for sku in range(1, 5)]
            self.assertEqual(canales_por_sku, list(repartir_entero(canales, perfil_inventario(demanda, txt_perfil), escalar_minimo=1)), txt_perfil)

    def test_basada_en_clases_agrupa_los_skus_ella_misma(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        config = dict(simulacion.config, vector_demanda_sku=np.array([0.05, 0.4, 0.1, 0.3, 0.15]), escalar_numero_skus=5,
                      texto_criterio_clases="skus", vector_proporcion_skus_por_clase=np.array([0.4, 0.6]))
        politica = crear_politica('basada_en_clases', E, G, P, config)
        self.assertEqual([politica.etiquetaZona(sku) for sku in range(1, 6)], [2, 1, 2, 1, 2])
        self.assertEqual(politica.matriz_zonificacion[0, 0, 0], 1)           # la clase de los SKUs más demandados queda más cerca


class PruebasClasesPorDemanda(unittest.TestCase):
    def test_partes_iguales_reparte_la_demanda_sin_indicar_cuantos_skus(self):
        from asrs.politicas.zonificacion import agrupar_skus_por_demanda
        from asrs.generacion.probabilidades import demanda_hausman
        demanda = demanda_hausman(30, 0.5)
        clase = agrupar_skus_por_demanda(demanda, [1, 1, 1])
        self.assertEqual(sorted(set(clase)), [1, 2, 3])
        for k in (1, 2, 3):
            self.assertAlmostEqual(float(demanda[clase == k].sum()), 1 / 3, delta=0.06)          # cada clase queda cerca de un tercio de la demanda
        self.assertTrue(np.all(np.diff(clase[np.argsort(-demanda, kind="stable")]) >= 0))        # la clase 1 son los más demandados
        self.assertLess(int((clase == 1).sum()), int((clase == 3).sum()))                         # pocos SKUs muy demandados y muchos poco demandados

    def test_pareto_da_80_15_5_de_la_demanda(self):
        from asrs.politicas.zonificacion import clasificar_skus
        from asrs.generacion.probabilidades import demanda_hausman
        demanda = demanda_hausman(30, 0.5)
        clase = clasificar_skus(demanda, {"texto_criterio_clases": "pareto"})
        self.assertEqual([round(float(demanda[clase == k].sum()), 2) for k in (1, 2, 3)], [0.80, 0.15, 0.05])

    def test_cada_clase_tiene_al_menos_un_sku_incluso_con_demanda_muy_concentrada(self):
        from asrs.politicas.zonificacion import agrupar_skus_por_demanda
        demanda = np.array([0.97, 0.01, 0.01, 0.005, 0.005])
        clase = agrupar_skus_por_demanda(demanda, [1, 1, 1, 1])
        self.assertEqual(sorted(set(clase)), [1, 2, 3, 4])

    def test_la_politica_usa_el_criterio_de_la_configuracion(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        por_defecto = crear_politica('basada_en_clases', E, G, P, simulacion.config)
        self.assertEqual(simulacion.config["texto_criterio_clases"], "skus")
        self.assertEqual(int(por_defecto.vector_clase_de_sku.max()), 3)
        # Con el criterio de partes iguales de la demanda, el número de clases es el configurado
        cinco = crear_politica('basada_en_clases', E, G, P, dict(simulacion.config, texto_criterio_clases="iguales", escalar_numero_clases=5))
        self.assertEqual(int(cinco.vector_clase_de_sku.max()), 5)


class PruebasRecuperacion(unittest.TestCase):
    def test_fcfs_elige_el_item_mas_antiguo_del_sku(self):
        simulacion = Simulacion()
        E, _, _ = simulacion.crearInstancias()
        E.matriz_contenedor_items[2, 2, 0] = Item(1)  # generación g
        E.matriz_contenedor_items[0, 5, 0] = Item(2)
        E.matriz_contenedor_items[1, 1, 0] = Item(1)  # generación mayor
        indice = PoliticaRecuperacionFCFS().seleccionar(E, 1)
        self.assertEqual(indice, (2, 2, 0))
        self.assertEqual(PoliticaRecuperacionFCFS().seleccionar(E, 3), ())

    def test_fcfs_accesible_prefiere_el_mas_antiguo_sin_nada_delante(self):
        from asrs.politicas.recuperacion import PoliticaRecuperacionFCFSAccesible
        simulacion = Simulacion()
        E, _, _ = simulacion.crearInstancias()
        E.matriz_contenedor_items[3, 3, 0] = Item(1)  # el más antiguo, al fondo
        E.matriz_contenedor_items[3, 3, 1] = Item(1)  # más nuevo, delante
        E.matriz_contenedor_items[5, 5, 0] = Item(1)  # el más nuevo, al fondo pero sin nada delante
        self.assertEqual(PoliticaRecuperacionFCFS().seleccionar(E, 1), (3, 3, 0))
        self.assertEqual(PoliticaRecuperacionFCFSAccesible().seleccionar(E, 1), (3, 3, 1))
        # si ninguna unidad del SKU está accesible, se retira la más antigua (y habrá que reubicar)
        E.matriz_contenedor_items[6, 6, 0] = Item(2)
        E.matriz_contenedor_items[6, 6, 1] = Item(3)
        self.assertEqual(PoliticaRecuperacionFCFSAccesible().seleccionar(E, 2), (6, 6, 0))

    def test_tras_reubicar_va_directo_a_la_celda_objetivo(self):
        from asrs.control.controlador_grua import ControladorGrua
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        objetivo, bloquea = (4, 6, 0), (4, 6, 1)
        E.matriz_contenedor_items[objetivo] = Item(1)
        E.matriz_contenedor_items[bloquea] = Item(2)
        random.seed(1)
        G.vector_eventos = []
        ControladorGrua(E, G, P, [(1, 1)], crear_politica('aleatoria', E, G, P, simulacion.config)).ejecutarSecuencia()

        eventos = G.vector_eventos
        indice_dejar = next(i for i, e in enumerate(eventos) if e[1] == 'D')          # la grúa deja el ítem que estorbaba
        indice_retirar = next(i for i, e in enumerate(eventos) if e[1] == 'P' and tuple(e[2:5]) == objetivo)
        destino_reubicacion = E.posicionesCeldas()[tuple(eventos[indice_dejar][2:5])][:2]
        self.assertTrue(np.any(destino_reubicacion != E.vector_posicion[:2]))     # lejos de la esquina, así volver a ella costaría tiempo
        movimientos = [e for e in eventos[indice_dejar + 1:indice_retirar] if e[1] == 'M' and e[2] > 0]
        # Sale del canal y va de la celda de reubicación a la celda objetivo (xy y luego z hasta el fondo), sin pasar por la esquina
        self.assertEqual(len(movimientos), 2)
        self.assertEqual(movimientos[1][3:5], [round(float(x), 4) for x in destino_reubicacion])
        self.assertEqual(movimientos[1][5:7], [round(float(x), 4) for x in E.posicionesCeldas()[objetivo][:2]])
        self.assertIsInstance(E.matriz_contenedor_items[objetivo], int)
    def test_fcfs_accesible_prefiere_el_item_sin_bloqueo(self):
        simulacion = Simulacion()
        E, _, _ = simulacion.crearInstancias()
        E.matriz_contenedor_items[2, 2, 0] = Item(1)  # el más antiguo, bloqueado
        E.matriz_contenedor_items[2, 2, 1] = Item(2)
        E.matriz_contenedor_items[1, 1, 0] = Item(1)  # más reciente, accesible
        self.assertEqual(PoliticaRecuperacionFCFS().seleccionar(E, 1), (2, 2, 0))
        self.assertEqual(PoliticaRecuperacionFCFSAccesible().seleccionar(E, 1), (1, 1, 0))
        # Si todos los ítems del SKU están bloqueados se elige el más antiguo, como en FCFS estricto
        E.matriz_contenedor_items[1, 1, 1] = Item(2)
        self.assertEqual(PoliticaRecuperacionFCFSAccesible().seleccionar(E, 1), (2, 2, 0))

    def test_se_cuentan_los_items_reubicados_no_las_tareas(self):
        from asrs.control.controlador_grua import ControladorGrua
        config = cargar_configuracion()
        config["vector_configuracion_estanteria"] = np.array([6, 6, 3])
        simulacion = Simulacion(config)
        E, G, P = simulacion.crearInstancias()
        politica = crear_politica('mas_cercana_es', E, G, P, config)
        E.matriz_contenedor_items[3, 3, 0] = Item(1)  # objetivo
        E.matriz_contenedor_items[3, 3, 1] = Item(2)  # dos bloqueadores
        E.matriz_contenedor_items[3, 3, 2] = Item(2)
        C = ControladorGrua(E, G, P, [], politica)
        C.recuperar((3, 3, 0))
        self.assertEqual(C.instancia_metricas.cantidad_reubicaciones, 2)
        self.assertEqual(C.instancia_metricas.cantidad_reubicaciones_recuperacion, 1)

    def test_tras_reubicar_la_grua_va_directo_al_objetivo(self):
        from asrs.control.controlador_grua import ControladorGrua
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        politica = crear_politica('mas_cercana_es', E, G, P, simulacion.config)
        # El canal de la esquina está lleno, así el bloqueador se reubica lejos de ella
        E.matriz_contenedor_items[0, 0, 0] = Item(3)
        E.matriz_contenedor_items[0, 0, 1] = Item(3)
        E.matriz_contenedor_items[5, 8, 0] = Item(1)  # objetivo
        E.matriz_contenedor_items[5, 8, 1] = Item(2)  # bloqueador
        C = ControladorGrua(E, G, P, [], politica)
        G.vector_eventos = []
        C.recuperar((5, 8, 0))
        eventos = G.vector_eventos
        i_soltar = next(i for i, e in enumerate(eventos) if e[1] == 'D')
        i_recoger = max(i for i, e in enumerate(eventos) if e[1] == 'P')
        traslados = [e for e in eventos[i_soltar:i_recoger] if e[1] == 'M' and (e[3], e[4]) != (e[5], e[6])]
        self.assertEqual(len(traslados), 1)
        destino = E.posicionesCeldas()[5, 8, 0]
        self.assertAlmostEqual(traslados[0][5], destino[0])
        self.assertAlmostEqual(traslados[0][6], destino[1])
        self.assertNotEqual((traslados[0][3], traslados[0][4]), (0.0, 0.0))


class PruebasEscenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulacion = Simulacion()
        cls.SA, cls.S = cls.simulacion.generarSecuencias()

    def test_largo_de_secuencias(self):
        self.assertEqual(len(self.S), self.simulacion.config["escalar_largo_secuencia"])

    def test_llenado_alcanza_la_ocupacion_fijada(self):
        for clave in POLITICAS_ASIGNACION:
            _, E, _, _, _ = preparar_estado(clave)
            esperado = int(E.escalar_capacidad_total * self.simulacion.config["escalar_nivel_ocupacion"])
            almacenado = np.count_nonzero(E.matriz_contenedor_items)
            if clave == 'full_turnover':
                # Con una zona fija por SKU y pocos canales por SKU, el redondeo a canales completos deja sin lugar a algunas unidades
                self.assertGreaterEqual(almacenado, 0.9 * esperado, clave)
                self.assertLessEqual(almacenado, esperado, clave)
            else:
                self.assertEqual(almacenado, esperado, clave)

    def test_politicas_con_zonas_respetan_su_zona(self):
        for clave in ('basada_en_clases', 'full_turnover'):
            _, E, _, _, politica = preparar_estado(clave)
            for indice in np.ndindex(E.matriz_contenedor_items.shape):
                item = E.matriz_contenedor_items[indice]
                if isinstance(item, Item):
                    self.assertEqual(politica.matriz_zonificacion[indice], politica.etiquetaZona(item.escalar_sku))

    def test_no_quedan_fondos_libres_tras_un_item_frontal(self):
        for clave in POLITICAS_ASIGNACION:
            simulacion, E, G, P, politica = preparar_estado(clave)
            from asrs.control.controlador_grua import ControladorGrua
            ControladorGrua(E, G, P, self.S, politica).ejecutarSecuencia()
            ocupacion = MatricesRestriccion().ocupacion(E)
            self.assertEqual(int((~ocupacion[:, :, 0] & ocupacion[:, :, 1]).sum()), 0, clave)

    def test_replica_reproducible_y_completa(self):
        config = cargar_configuracion()
        config["vector_politicas_recuperacion"] = ('fcfs', 'fcfs_accesible')
        r1 = Simulacion.cargarReplica((config, 7))
        r2 = Simulacion.cargarReplica((config, 7))
        self.assertEqual(r1, r2)
        self.assertEqual(len(r1), len(config["vector_politicas_asignacion"]) * len(config["vector_politicas_recuperacion"]))
        self.assertEqual(len(r1[0]), 31)
        self.assertEqual([fila[0] for fila in r1[::2]],
                         ['Aleatoria', 'BasadaEnClases', 'FullTurnover', 'MasCercanaAlES', 'MasCercanaALaGrua'])
        self.assertEqual({(fila[1]) for fila in r1}, {'FCFS', 'FCFS-Accesible'})

    def test_fcfs_entre_accesibles_reubica_menos(self):
        # Con la misma asignación y el mismo azar, elegir entre los accesibles nunca debería reubicar más que FCFS estricto
        config = cargar_configuracion()
        config["vector_politicas_recuperacion"] = ('fcfs', 'fcfs_accesible')
        filas = Simulacion.cargarReplica((config, 7))
        for estricto, accesible in zip(filas[::2], filas[1::2]):
            self.assertEqual(estricto[0], accesible[0])
            self.assertLessEqual(accesible[8], estricto[8], estricto[0])


class PruebasVisualizacion(unittest.TestCase):
    def test_reloj_grabado_coincide_con_las_metricas(self):
        from asrs.visualizacion.generador import generar_datos
        config = cargar_configuracion()
        datos = generar_datos(config, [2024])
        replica = datos["replicas"][0]
        self.assertEqual(len(replica["politicas"]), len(config["vector_politicas_asignacion"]) * len(config["vector_politicas_recuperacion"]))
        for politica in replica["politicas"]:
            self.assertAlmostEqual(politica["reloj_final"], politica["tiempo_total"], places=2, msg=politica["nombre"])
            self.assertEqual(politica["eventos"][-1][1], 'F')
            self.assertEqual(sum(1 for e in politica["eventos"] if e[1] == 'T'), len(replica["secuencia"]))


class PruebasConfiguracion(unittest.TestCase):
    def test_ida_y_vuelta_conserva_los_valores(self):
        from asrs.config import config_a_json, config_desde_json
        original = cargar_configuracion()
        copia = config_desde_json(config_a_json(original))
        self.assertEqual(copia["vector_configuracion_estanteria"].tolist(), [18, 20, 2])
        self.assertEqual(copia["escalar_largo_secuencia"], original["escalar_largo_secuencia"])
        self.assertEqual(copia["vector_politicas_asignacion"], original["vector_politicas_asignacion"])

    def test_demanda_personalizada_se_normaliza_y_fija_la_cantidad_de_skus(self):
        from asrs.config import config_desde_json
        config = config_desde_json({"vector_demanda_sku_personalizada": [5, 5, 10]})
        self.assertEqual(config["vector_demanda_sku"].tolist(), [0.25, 0.25, 0.5])
        self.assertEqual(config["escalar_numero_skus"], 3)

    def test_demanda_de_hausman_por_defecto(self):
        from asrs.config import config_desde_json
        config = config_desde_json({"escalar_numero_skus": 12, "escalar_sigma_hausman": 0.6})
        self.assertEqual(len(config["vector_demanda_sku"]), 12)
        self.assertAlmostEqual(float(config["vector_demanda_sku"].sum()), 1.0)

    def test_aceleracion_por_defecto_y_validacion(self):
        from asrs.config import config_desde_json
        config = config_desde_json({})
        # Instancia del capítulo 6: 0,5 m/s² en filas y columnas y velocidad constante en la profundidad
        self.assertEqual(config["vector_aceleracion_grua"].tolist(), [0.5, 0.5, 0.0])
        self.assertEqual(config_desde_json({"vector_aceleracion_grua": [0, 0, 0]})["vector_aceleracion_grua"].tolist(), [0, 0, 0])
        for caso in ({"vector_aceleracion_grua": [1, -1, 1]}, {"vector_deceleracion_grua": [1, 1]}, {"vector_aceleracion_grua": [1, "x", 1]}):
            with self.assertRaises(ValueError, msg=str(caso)):
                config_desde_json(caso)

    def test_tareas_por_defecto_son_el_diez_por_ciento(self):
        from asrs.config import config_desde_json
        self.assertEqual(config_desde_json({"vector_configuracion_estanteria": [10, 10, 2], "escalar_numero_skus": 20})["escalar_largo_secuencia"], 20)

    def test_rechaza_valores_invalidos(self):
        from asrs.config import config_desde_json
        invalidos = [
            {"vector_configuracion_estanteria": [0, 5, 2]},
            {"vector_configuracion_estanteria": [100, 100, 2]},
            {"vector_cotas_estanteria": [1, -1, 1]},
            {"vector_velocidad_grua": [1, 0, 1]},
            {"escalar_nivel_ocupacion": 1.5},
            {"escalar_numero_skus": 0},
            {"escalar_numero_skus": 500},
            {"escalar_sigma_hausman": 1.5},
            {"vector_demanda_sku_personalizada": [0, 0]},
            {"vector_demanda_sku_personalizada": [1, -1]},
            {"texto_criterio_clases": "skus", "escalar_numero_skus": 3, "vector_proporcion_skus_por_clase": [1, 1, 1, 1]},
            {"texto_criterio_clases": "iguales", "escalar_numero_skus": 3, "escalar_numero_clases": 4},
            {"texto_criterio_clases": "otro"},
            {"escalar_numero_clases": 0},
            {"vector_proporcion_skus_por_clase": [1, 0, 1]},
            {"vector_configuracion_estanteria": [2, 2, 2], "escalar_numero_skus": 30},
            {"vector_politicas_asignacion": []},
            {"vector_politicas_asignacion": ["inexistente"]},
            {"escalar_semilla": "abc"},
            {"vector_prob_acum_AR": [0.5, 0.9]},
            {"texto_perfil_inventario": "otro"},
            {"texto_regla_recuperacion": "lifo"},
            {"vector_politicas_recuperacion": []},
            {"vector_politicas_recuperacion": ["fcfs", "fcfs"]},
        ]
        for caso in invalidos:
            with self.assertRaises(ValueError, msg=str(caso)):
                config_desde_json(caso)

    def test_regla_de_recuperacion_antigua_sigue_cargando(self):
        from asrs.config import config_desde_json
        # Las configuraciones guardadas antes de admitir varias recuperaciones traían una sola en 'texto_regla_recuperacion'
        self.assertEqual(config_desde_json({})["vector_politicas_recuperacion"], ('fcfs',))
        self.assertEqual(config_desde_json({"texto_regla_recuperacion": "fcfs_accesible"})["vector_politicas_recuperacion"], ('fcfs_accesible',))
        self.assertEqual(config_desde_json({"texto_regla_recuperacion": "fcfs_accesible", "vector_politicas_recuperacion": ["fcfs"]})["vector_politicas_recuperacion"], ('fcfs',))


class PruebasEstadistica(unittest.TestCase):
    def test_resumen_con_victorias_pareadas(self):
        from asrs.simulacion.estadistica import resumir

        def fila(nombre, total):
            return [nombre, 'FCFS'] + [0] * 27 + [10, total]

        resultado = resumir([[fila("A", 100), fila("B", 200)], [fila("A", 300), fila("B", 200)]])
        a, b = resultado["politicas"]
        self.assertEqual(a["recuperacion"], 'FCFS')
        self.assertEqual(resultado["replicas"], 2)
        self.assertEqual((a["victorias"], b["victorias"]), (1, 1))
        self.assertEqual(a["tiempo_total"]["media"], 200)
        self.assertEqual(b["tiempo_total"]["desviacion"], 0)
        self.assertEqual(a["tiempo_total"]["minimo"], 100)

    def test_cuantil_t_de_student(self):
        from asrs.simulacion.estadistica import cuantil_t_975
        # Valores de tabla de la t de Student para el 95 % bilateral
        for escalar_gl, escalar_tabla in ((1, 12.706), (2, 4.303), (3, 3.182), (10, 2.228), (19, 2.093), (30, 2.042), (1000, 1.962)):
            self.assertAlmostEqual(cuantil_t_975(escalar_gl) / escalar_tabla, 1, delta=0.002, msg=str(escalar_gl))

    def test_comparacion_pareada_con_la_mejor(self):
        from asrs.simulacion.estadistica import resumir

        def fila(nombre, total):
            return [nombre] + [0] * 27 + [10, total]

        # B tarda siempre 10 s más que A: los intervalos de cada una se solapan, pero la diferencia réplica a réplica es constante
        resultado = resumir([[fila("A", total), fila("B", total + 10)] for total in (100, 300, 200)])
        a, b = resultado["politicas"]
        self.assertIsNone(a["tiempo_total"]["vs_mejor"])
        self.assertEqual(b["tiempo_total"]["vs_mejor"]["media"], 10)
        self.assertEqual(b["tiempo_total"]["vs_mejor"]["ic95"], 0)
        self.assertGreater(a["tiempo_total"]["media"] + a["tiempo_total"]["ic95"], b["tiempo_total"]["media"] - b["tiempo_total"]["ic95"])


class PruebasServidor(unittest.TestCase):
    def test_la_solicitud_admite_el_maximo_de_tareas(self):
        import json
        from asrs.config import LIMITE_TAREAS
        from asrs.visualizacion.servidor import LIMITE_CUERPO
        txt_csv = "SKU;Task\n" + "".join(f"PROD-{i % 200:05d};{'A' if i % 2 else 'R'}\n" for i in range(LIMITE_TAREAS))
        self.assertLess(len(json.dumps({"config": {"texto_tareas_csv": txt_csv}}).encode("utf-8")), LIMITE_CUERPO)

    def test_un_segundo_servidor_en_el_mismo_puerto_falla(self):
        from asrs.visualizacion.servidor import crear_servidor
        primero = crear_servidor(0)
        try:
            with self.assertRaises(OSError):
                crear_servidor(primero.server_address[1])
        finally:
            primero.server_close()

    def test_simulacion_completa_por_http(self):
        import json
        import threading
        import time
        import urllib.error
        import urllib.request

        from asrs.visualizacion.servidor import crear_servidor

        servidor = crear_servidor(0)
        hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
        hilo.start()
        base = f"http://127.0.0.1:{servidor.server_address[1]}"
        try:
            def pedir(ruta, cuerpo=None):
                solicitud = urllib.request.Request(base + ruta, data=None if cuerpo is None else json.dumps(cuerpo).encode(),
                                                   headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(solicitud, timeout=60) as respuesta:
                        return respuesta.status, json.loads(respuesta.read())
                except urllib.error.HTTPError as error:
                    return error.code, json.loads(error.read())

            self.assertEqual(pedir("/api/activo"), (200, {"id": None}))
            codigo, defecto = pedir("/api/config")
            self.assertEqual(codigo, 200)
            self.assertEqual(defecto["vector_configuracion_estanteria"], [18, 20, 2])

            codigo, error = pedir("/api/simular", {"config": {"vector_velocidad_grua": [1, 0, 1]}})
            self.assertEqual(codigo, 400)
            self.assertIn("Velocidad", error["error"])

            pequena = {"vector_configuracion_estanteria": [4, 5, 2], "escalar_numero_skus": 6, "escalar_largo_secuencia": 6, "escalar_nivel_ocupacion": 0.5,
                       "vector_politicas_asignacion": ["aleatoria", "mas_cercana_es"], "vector_politicas_recuperacion": ["fcfs", "fcfs_accesible"], "escalar_replicas_animadas": 1, "escalar_cantidad_replicas": 2}
            codigo, creado = pedir("/api/simular", {"config": pequena})
            self.assertEqual(codigo, 202)
            limite = time.time() + 60
            while time.time() < limite:
                _, estado = pedir("/api/trabajo/" + creado["id"])
                if estado["estado"] in ("listo", "error"):
                    break
                time.sleep(0.2)
            self.assertEqual(estado["estado"], "listo", estado.get("error"))
            _, completo = pedir("/api/trabajo/" + creado["id"] + "?resultado=1")
            resultado = completo["resultado"]
            self.assertEqual(len(resultado["datos"]["replicas"]), 1)
            self.assertEqual(resultado["datos"]["configuracion"], [4, 5, 2])
            self.assertEqual(resultado["estadistica"]["replicas"], 2)
            self.assertEqual(len(resultado["estadistica"]["politicas"]), 4)
        finally:
            servidor.shutdown()
            servidor.server_close()


if __name__ == "__main__":
    unittest.main()
