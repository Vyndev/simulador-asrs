"""Pruebas de regresión del simulador ASRS (politicas). Ejecutar con: python -m unittest discover -s tests"""
import numpy as np
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
from asrs.config import cargar_configuracion
from asrs.dominio.item import Item
from asrs.politicas.recuperacion import PoliticaRecuperacionFCFS, PoliticaRecuperacionFCFSAccesible
from asrs.politicas.registro import POLITICAS_ASIGNACION, crear_politica
from asrs.politicas.restriccion import MatricesRestriccion
from asrs.politicas.zonificacion import agrupar_skus_en_clases, construir_zonificacion
from asrs.simulacion.simulacion import Simulacion


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


if __name__ == "__main__":
    unittest.main()
