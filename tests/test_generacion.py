"""Pruebas de regresión del simulador ASRS (generacion). Ejecutar con: python -m unittest discover -s tests"""
import numpy as np
import random
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
from asrs.config import cargar_configuracion
from asrs.simulacion.simulacion import Simulacion


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
        self.assertEqual(len(resultado), len(cfg["vector_politicas_asignacion"]) * len(cfg["vector_politicas_recuperacion"]))

    def test_datos_invalidos_se_rechazan_en_la_configuracion(self):
        from asrs.config import config_a_json, config_desde_json
        datos = config_a_json(cargar_configuracion())
        with self.assertRaises(ValueError):
            config_desde_json({**datos, "texto_origen_tareas": "datos", "texto_tareas_csv": ""})
        with self.assertRaises(ValueError):
            config_desde_json({**datos, "texto_origen_tareas": "otro"})


if __name__ == "__main__":
    unittest.main()
