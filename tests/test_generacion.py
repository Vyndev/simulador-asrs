"""Pruebas de regresión del simulador ASRS (generacion). Ejecutar con: python -m unittest discover -s tests"""
import numpy as np
import random
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
from auxiliar import TAREAS_PRUEBA, Simulacion, cargar_configuracion
from asrs.politicas.registro import POLITICAS_ASIGNACION


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

    def test_el_llenado_sigue_el_inventario_del_calentamiento(self):
        # El llenado es el inventario que el modelo lleva al final del calentamiento: cada SKU aparece tantas veces como unidades tiene
        for txt_modelo in ("ciclo_vida", "estatica"):
            simulacion = Simulacion(dict(cargar_configuracion(), texto_modelo_demanda=txt_modelo))
            random.seed(3)
            SA, S = simulacion.generarSecuencias()
            cfg = simulacion.config
            vector_inventario_inicial = simulacion.escenario_demanda.vector_inventario_inicial
            conteo = [sum(1 for _, sku in SA if sku == i) for i in range(1, len(vector_inventario_inicial) + 1)]
            self.assertEqual(conteo, vector_inventario_inicial.tolist(), txt_modelo)
            self.assertGreater(len(SA), 0)

    def test_generar_secuencias_usa_el_escenario(self):
        simulacion = Simulacion()
        random.seed(5)
        SA, S = simulacion.generarSecuencias()
        self.assertIsNotNone(simulacion.escenario_demanda)
        self.assertEqual(SA, simulacion.escenario_demanda.lista_llenado)
        self.assertEqual(S, simulacion.escenario_demanda.lista_operacion)
        self.assertEqual(len(S), TAREAS_PRUEBA)
        # La misma semilla da el mismo escenario
        otra = Simulacion()
        random.seed(5)
        self.assertEqual(otra.generarSecuencias(), (SA, S))

    def test_argumentos_de_operacion_con_escenario(self):
        simulacion = Simulacion()
        self.assertIsNone(simulacion.escenario_demanda)
        self.assertEqual(simulacion.argumentosOperacion(10), {})
        self.assertIsNone(simulacion.configPolitica()["escenario_demanda"])
        random.seed(5)
        simulacion.generarSecuencias()
        escenario = simulacion.escenario_demanda
        self.assertIs(simulacion.configPolitica()["escenario_demanda"], escenario)
        self.assertNotIn("escenario_demanda", simulacion.config)    # la configuración original no se modifica
        argumentos = simulacion.argumentosOperacion(TAREAS_PRUEBA)
        self.assertEqual(argumentos["vector_dia_de_tarea"], escenario.vector_dia_de_tarea[:TAREAS_PRUEBA])
        self.assertEqual(argumentos["conjunto_retiros"], escenario.conjunto_retiros)
        corto = simulacion.argumentosOperacion(5)
        self.assertEqual(len(corto["vector_dia_de_tarea"]), 5)
        self.assertEqual(corto["conjunto_retiros"], {i for i in escenario.conjunto_retiros if i < 5})

    def test_argumentos_de_operacion_recortan_los_retiros(self):
        # Horizonte largo y ciclo de un mes: hay cientos de retiros repartidos por toda la lista
        simulacion = Simulacion(dict(cargar_configuracion(), escalar_largo_secuencia=90 * 1440, escalar_meses_ciclo_vida=1))
        random.seed(5)
        simulacion.generarSecuencias()
        conjunto_total = simulacion.escenario_demanda.conjunto_retiros
        escalar_n = 45 * 1440
        self.assertGreater(len(conjunto_total), 100)
        argumentos = simulacion.argumentosOperacion(escalar_n)
        conjunto_corto = argumentos["conjunto_retiros"]
        self.assertGreater(len(conjunto_corto), 0)
        self.assertLess(len(conjunto_corto), len(conjunto_total))
        self.assertTrue(all(i < escalar_n for i in conjunto_corto))
        self.assertEqual(len(conjunto_corto), sum(1 for i in conjunto_total if i < escalar_n))
        self.assertEqual(len(argumentos["vector_dia_de_tarea"]), escalar_n)
        self.assertEqual(argumentos["vector_dia_de_tarea"], simulacion.escenario_demanda.vector_dia_de_tarea[:escalar_n])


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
        self.assertEqual(cfg["escalar_largo_secuencia"], 3)          # 7 tareas, historial del 50 % (4): se evalúan las 3 siguientes
        self.assertEqual(cfg["escalar_numero_skus"], 3)
        random.seed(1)
        _, tareas = Simulacion(cfg).generarSecuencias()
        self.assertEqual(tareas, [tuple(t) for t in cfg["secuencia_real"]])
        resultado = Simulacion.cargarReplica((cfg, 1))
        self.assertEqual(len(resultado), len(cfg["vector_politicas_asignacion"]) * len(cfg["vector_politicas_recuperacion"]))

    def test_datos_reales_sin_escenario(self):
        from asrs.config import config_a_json, config_desde_json
        datos = config_a_json(cargar_configuracion())
        datos.update(texto_origen_tareas="datos", texto_tareas_csv=self.CSV, escalar_porcentaje_historial=50)
        simulacion = Simulacion(config_desde_json(datos))
        random.seed(1)
        simulacion.generarSecuencias()
        self.assertIsNone(simulacion.escenario_demanda)
        self.assertEqual(simulacion.argumentosOperacion(10), {})
        self.assertIsNone(simulacion.configPolitica()["escenario_demanda"])
        # La réplica con datos reales solo informa la semilla y el controlador no cuenta retiros
        filas, dic_resumen = Simulacion.cargarReplicaConEscenario((simulacion.config, 1))
        self.assertEqual(dic_resumen, {"semilla": 1})
        # Cant_RetirosFinCiclo es la cuarta columna desde el final de la fila
        self.assertTrue(all(fila[-4] == 0 for fila in filas))

    def config_real_larga(self):
        # Lista real de 400 tareas sobre 12 SKUs, con muchos almacenamientos y recuperaciones intercalados para que los desempates al azar importen
        from asrs.config import config_a_json, config_desde_json
        generador = np.random.default_rng(3)
        vector_filas = ["SKU;Task"] + [f"P{generador.integers(1, 13)};{'A' if i % 2 == 0 else 'R'}" for i in range(400)]
        datos = config_a_json(cargar_configuracion())
        datos.update(texto_origen_tareas="datos", texto_tareas_csv="\n".join(vector_filas), escalar_porcentaje_historial=50)
        return config_desde_json(datos)

    def test_la_semilla_de_una_politica_no_depende_de_las_demas_elegidas(self):
        config = self.config_real_larga()
        config["vector_politicas_asignacion"] = tuple(POLITICAS_ASIGNACION)
        dic_completo = {fila[0]: fila for fila in Simulacion.cargarReplica((config, 3))}
        for vector_subconjunto in (('full_turnover', 'aleatoria'), ('mas_cercana_grua', 'basada_en_clases')):
            config["vector_politicas_asignacion"] = vector_subconjunto
            for fila in Simulacion.cargarReplica((config, 3)):
                self.assertEqual(fila, dic_completo[fila[0]], vector_subconjunto)
        # Es reproducible: repetir la réplica da lo mismo
        config["vector_politicas_asignacion"] = tuple(POLITICAS_ASIGNACION)
        self.assertEqual({fila[0]: fila for fila in Simulacion.cargarReplica((config, 3))}, dic_completo)
        # Cada política tiene la semilla de su posición en el registro
        self.assertEqual(len({Simulacion.semillaDePolitica(3, txt_clave) for txt_clave in ('aleatoria', 'basada_en_clases', 'full_turnover',
                                                                                              'mas_cercana_es', 'mas_cercana_grua')}), 5)

    def test_datos_invalidos_se_rechazan_en_la_configuracion(self):
        from asrs.config import config_a_json, config_desde_json
        datos = config_a_json(cargar_configuracion())
        with self.assertRaises(ValueError):
            config_desde_json({**datos, "texto_origen_tareas": "datos", "texto_tareas_csv": ""})
        with self.assertRaises(ValueError):
            config_desde_json({**datos, "texto_origen_tareas": "otro"})


if __name__ == "__main__":
    unittest.main()
