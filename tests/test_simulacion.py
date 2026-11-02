"""Pruebas de regresión del simulador ASRS (simulacion). Ejecutar con: python -m unittest discover -s tests"""
import numpy as np
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
from asrs.config import cargar_configuracion
from asrs.dominio.item import Item
from asrs.politicas.registro import POLITICAS_ASIGNACION
from asrs.politicas.restriccion import MatricesRestriccion
from asrs.simulacion.simulacion import Simulacion
from auxiliar import preparar_estado


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
        self.assertEqual(len(r1), len(config["vector_politicas_asignacion"]) * len(config["vector_politicas_recuperacion"]))
        self.assertEqual(len(r1[0]), 31)
        self.assertEqual([fila[0] for fila in r1[::2]],
                         ['Aleatoria', 'BasadaEnClases', 'FullTurnover', 'MasCercanaAlES', 'MasCercanaALaGrua'])
        self.assertEqual({(fila[1]) for fila in r1}, {'FCFS', 'FCFS-Accesible'})

    def test_fcfs_entre_accesibles_reubica_menos(self):
        # Con la misma asignación y el mismo azar, elegir entre los accesibles nunca debería reubicar más que FCFS estricto
        config = cargar_configuracion()
        filas = Simulacion.cargarReplica((config, 7))
        for estricto, accesible in zip(filas[::2], filas[1::2]):
            self.assertEqual(estricto[0], accesible[0])
            self.assertLessEqual(accesible[8], estricto[8], estricto[0])


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
            return [nombre, 'FCFS'] + [0] * 27 + [10, total]

        resultado = resumir([[fila("A", 100), fila("B", 200)], [fila("A", 300), fila("B", 200)]])
        a, b = resultado["politicas"]
        self.assertEqual(a["recuperacion"], 'FCFS')
        self.assertEqual(resultado["replicas"], 2)
        self.assertEqual((a["victorias"], b["victorias"]), (1, 1))
        self.assertEqual(a["tiempo_total"]["media"], 200)
        self.assertEqual(b["tiempo_total"]["desviacion"], 0)
        self.assertEqual(a["tiempo_total"]["minimo"], 100)


if __name__ == "__main__":
    unittest.main()
