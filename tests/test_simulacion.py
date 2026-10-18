"""Pruebas de regresión del simulador ASRS. Ejecutar con: python -m unittest discover -s tests"""
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asrs.config import cargar_configuracion
from asrs.dominio.grua import Grua
from asrs.dominio.item import Item
from asrs.politicas.recuperacion import PoliticaRecuperacionFCFS
from asrs.politicas.registro import POLITICAS_ASIGNACION, crear_politica
from asrs.politicas.restriccion import MatricesRestriccion
from asrs.politicas.zonificacion import agrupar_skus_en_clases, construir_zonificacion
from asrs.simulacion.simulacion import Simulacion


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

    def test_tiempo_chebyshev_toma_el_eje_mas_lento(self):
        from asrs.cinematica import tiempo_chebyshev_xy
        self.assertAlmostEqual(float(tiempo_chebyshev_xy(np.array([2.0, 10.0]), np.array([2.0, 2.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))), 7.0)


class PruebasGrua(unittest.TestCase):
    def setUp(self):
        self.E, self.G, self.P = Simulacion().crearInstancias()

    def test_volver_a_plano_cuenta_tiempo(self):
        grua = Grua(np.array([0, 0, 2.7]), np.array([1, 1.5, 0.7]), 0)
        grua.irAEsquinaEstanteria(self.E)
        grua.moverEnEstanteria((3, 3, 0))
        self.assertGreater(grua.volverAplano(), 0)

    def test_la_aceleracion_alarga_los_desplazamientos(self):
        sin = Grua(np.array([0, 0, 2.7]), np.array([1, 1.5, 0.7]), 0)
        con = Grua(np.array([0, 0, 2.7]), np.array([1, 1.5, 0.7]), 0, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
        origen, destino = np.array([0.0, 0.0, 2.7]), np.array([9.6, 12.6, 0.0])
        self.assertAlmostEqual(sin.intancia_metricas.tiempoDezplamiento(origen, destino, sin.vector_velocidad), max(9.6 / 1.0, 12.6 / 1.5) + 2.7 / 0.7)
        self.assertGreater(con._tiempo(origen, destino), sin._tiempo(origen, destino))

    def test_posicion_riel_no_se_trunca(self):
        grua = Grua(np.array([0, 0, 2.7]), np.array([1, 1.5, 0.7]), 0)
        grua.irAEsquinaEstanteria(self.E)
        self.assertAlmostEqual(grua.vector_posicion_actual[2], 2.7)


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
        self.assertEqual(len(resultado), len(cfg["vector_politicas_asignacion"]))

    def test_datos_invalidos_se_rechazan_en_la_configuracion(self):
        from asrs.config import config_a_json, config_desde_json
        datos = config_a_json(cargar_configuracion())
        with self.assertRaises(ValueError):
            config_desde_json({**datos, "texto_origen_tareas": "datos", "texto_tareas_csv": ""})
        with self.assertRaises(ValueError):
            config_desde_json({**datos, "texto_origen_tareas": "otro"})


try:
    import gymnasium  # noqa: F401
    HAY_GYMNASIUM = True
except ImportError:
    HAY_GYMNASIUM = False


@unittest.skipUnless(HAY_GYMNASIUM, "faltan las librerías de aprendizaje por refuerzo (requirements-rl.txt)")
class PruebasEntornoRL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = cargar_configuracion()
        cls.config["vector_configuracion_estanteria"] = np.array([6, 8, 2])
        cls.config["escalar_largo_secuencia"] = 20

    def _episodio(self, entorno, escalar_estado, elegir):
        entorno.reset(options={"estado": escalar_estado})
        retorno, terminado = 0.0, entorno.escalar_indice >= len(entorno.S)
        while not terminado:
            _, recompensa, terminado, _, _ = entorno.step(elegir(entorno))
            retorno += recompensa
        return retorno

    def test_el_retorno_es_el_tiempo_total_y_es_reproducible(self):
        from asrs.rl.entorno import ESCALAR_SEGUNDOS_POR_UNIDAD, EntornoAsignacion
        entorno = EntornoAsignacion(self.config, 77, 2)
        primero = lambda e: int(np.flatnonzero(e.action_masks())[0])
        retorno = self._episodio(entorno, 0, primero)
        self.assertAlmostEqual(-retorno * ESCALAR_SEGUNDOS_POR_UNIDAD, entorno.tiempo_total(), places=3)
        self.assertEqual(self._episodio(entorno, 0, primero), retorno)

    def test_acciones_invalidas_se_penalizan_y_no_rompen(self):
        from asrs.rl.entorno import EntornoAsignacion
        entorno = EntornoAsignacion(self.config, 77, 2)
        entorno.reset(options={"estado": 1})
        vector_invalidos = np.flatnonzero(~entorno.action_masks())
        if not len(vector_invalidos):
            self.skipTest("todos los canales tienen espacio")
        _, recompensa, _, _, _ = entorno.step(int(vector_invalidos[0]))
        self.assertEqual(entorno.escalar_invalidas, 1)
        self.assertLess(recompensa, -0.99)

    def test_las_clasicas_parten_del_mismo_estado(self):
        from asrs.rl.evaluacion import evaluar_politica_clasica
        a = evaluar_politica_clasica(self.config, "mas_cercana_es", 77, [0, 1])["tiempos"]
        b = evaluar_politica_clasica(self.config, "mas_cercana_es", 77, [0, 1])["tiempos"]
        self.assertTrue(np.allclose(a, b))


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

    def test_basada_en_clases_agrupa_los_skus_ella_misma(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        config = dict(simulacion.config, vector_demanda_sku=np.array([0.05, 0.4, 0.1, 0.3, 0.15]), escalar_numero_skus=5,
                      vector_proporcion_skus_por_clase=np.array([0.4, 0.6]))
        politica = crear_politica('basada_en_clases', E, G, P, config)
        self.assertEqual([politica.etiquetaZona(sku) for sku in range(1, 6)], [2, 1, 2, 1, 2])
        self.assertEqual(politica.matriz_zonificacion[0, 0, 0], 1)           # la clase de los SKUs más demandados queda más cerca


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
            self.assertEqual(np.count_nonzero(E.matriz_contenedor_items), esperado, clave)

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
        r1 = Simulacion.cargarReplica((config, 7))
        r2 = Simulacion.cargarReplica((config, 7))
        self.assertEqual(r1, r2)
        self.assertEqual(len(r1), len(config["vector_politicas_asignacion"]))
        self.assertEqual(len(r1[0]), 30)
        self.assertEqual([fila[0] for fila in r1],
                         ['Aleatoria', 'BasadaEnClases', 'FullTurnover', 'MasCercanaAlES', 'MasCercanaALaGrua'])


class PruebasVisualizacion(unittest.TestCase):
    def test_reloj_grabado_coincide_con_las_metricas(self):
        from asrs.visualizacion.generador import generar_datos
        datos = generar_datos(cargar_configuracion(), [2024])
        replica = datos["replicas"][0]
        self.assertEqual(len(replica["politicas"]), 5)
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
        self.assertEqual(config["vector_aceleracion_grua"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(config_desde_json({"vector_aceleracion_grua": [0, 0, 0]})["vector_aceleracion_grua"].tolist(), [0, 0, 0])
        for caso in ({"vector_aceleracion_grua": [1, -1, 1]}, {"vector_deceleracion_grua": [1, 1]}, {"vector_aceleracion_grua": [1, "x", 1]}):
            with self.assertRaises(ValueError, msg=str(caso)):
                config_desde_json(caso)

    def test_tareas_por_defecto_son_el_diez_por_ciento(self):
        from asrs.config import config_desde_json
        self.assertEqual(config_desde_json({"vector_configuracion_estanteria": [10, 10, 2]})["escalar_largo_secuencia"], 20)

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
            {"escalar_numero_skus": 3, "vector_proporcion_skus_por_clase": [1, 1, 1, 1]},
            {"vector_proporcion_skus_por_clase": [1, 0, 1]},
            {"vector_configuracion_estanteria": [2, 2, 2], "escalar_numero_skus": 30},
            {"vector_politicas_asignacion": []},
            {"vector_politicas_asignacion": ["inexistente"]},
            {"escalar_semilla": "abc"},
            {"vector_prob_acum_AR": [0.5, 0.9]},
        ]
        for caso in invalidos:
            with self.assertRaises(ValueError, msg=str(caso)):
                config_desde_json(caso)


class PruebasEstadistica(unittest.TestCase):
    def test_resumen_con_victorias_pareadas(self):
        from asrs.simulacion.estadistica import resumir

        def fila(nombre, total):
            return [nombre] + [0] * 27 + [10, total]

        resultado = resumir([[fila("A", 100), fila("B", 200)], [fila("A", 300), fila("B", 200)]])
        a, b = resultado["politicas"]
        self.assertEqual(resultado["replicas"], 2)
        self.assertEqual((a["victorias"], b["victorias"]), (1, 1))
        self.assertEqual(a["tiempo_total"]["media"], 200)
        self.assertEqual(b["tiempo_total"]["desviacion"], 0)
        self.assertEqual(a["tiempo_total"]["minimo"], 100)


class PruebasServidor(unittest.TestCase):
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
                       "vector_politicas_asignacion": ["aleatoria", "mas_cercana_es"], "escalar_replicas_animadas": 1, "escalar_cantidad_replicas": 2}
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
            self.assertEqual(len(resultado["estadistica"]["politicas"]), 2)
        finally:
            servidor.shutdown()
            servidor.server_close()


if __name__ == "__main__":
    unittest.main()
