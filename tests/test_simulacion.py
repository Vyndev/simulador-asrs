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
from asrs.politicas.zonificacion import construir_zonificacion
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


class PruebasGrua(unittest.TestCase):
    def setUp(self):
        self.E, self.G, self.P = Simulacion().crearInstancias()

    def test_volver_a_plano_cuenta_tiempo(self):
        grua = Grua(np.array([0, 0, 2.7]), np.array([1, 1.5, 0.7]), 0)
        grua.irAEsquinaEstanteria(self.E)
        grua.moverEnEstanteria((3, 3, 0))
        self.assertGreater(grua.volverAplano(), 0)

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
    def test_zonas_con_tamano_y_orden(self):
        tiempos = np.arange(8, dtype=float).reshape(2, 2, 2)
        zonas = construir_zonificacion(tiempos, [0.25, 0.75], [2, 1])
        # las proporciones se indexan por etiqueta: la clase 1 tiene 25 % y la clase 2 tiene 75 %
        self.assertEqual(int((zonas == 1).sum()), 2)
        self.assertEqual(int((zonas == 2).sum()), 6)
        # la clase 2 va primera en el orden, así que recibe las celdas de menor tiempo
        self.assertTrue(np.all(tiempos[zonas == 2] < tiempos[zonas == 1].min()))


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

    def test_full_turnover_ordena_por_demanda(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        config = dict(simulacion.config, vector_demanda_clase=np.array([0.1, 0.2, 0.7]))
        politica = crear_politica('full_turnover', E, G, P, config)
        self.assertEqual(politica.ordenClases(), [3, 2, 1])
        self.assertEqual(politica.matriz_zonificacion[0, 0, 0], 3)


class PruebasRecuperacion(unittest.TestCase):
    def test_fcfs_elige_el_item_mas_antiguo_de_la_clase(self):
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
                    self.assertEqual(politica.matriz_zonificacion[indice], item.escalar_etiqueta_clase)

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

    def test_normaliza_capacidad_y_demanda(self):
        from asrs.config import config_desde_json
        config = config_desde_json({"vector_capacidad_clase": [1, 1, 2], "vector_demanda_clase": [5, 5, 10]})
        self.assertAlmostEqual(float(config["vector_capacidad_clase"].sum()), 1.0)
        self.assertEqual(config["vector_demanda_clase"].tolist(), [0.25, 0.25, 0.5])

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
            {"vector_capacidad_clase": [0.5, 0.5], "vector_demanda_clase": [1]},
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

            codigo, defecto = pedir("/api/config")
            self.assertEqual(codigo, 200)
            self.assertEqual(defecto["vector_configuracion_estanteria"], [18, 20, 2])

            codigo, error = pedir("/api/simular", {"config": {"vector_velocidad_grua": [1, 0, 1]}})
            self.assertEqual(codigo, 400)
            self.assertIn("Velocidad", error["error"])

            pequena = {"vector_configuracion_estanteria": [4, 5, 2], "escalar_largo_secuencia": 6, "escalar_nivel_ocupacion": 0.5,
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
