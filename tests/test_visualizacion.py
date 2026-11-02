"""Pruebas de regresión del simulador ASRS (visualizacion). Ejecutar con: python -m unittest discover -s tests"""
import random
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
import test_generacion
from auxiliar import TAREAS_PRUEBA, Simulacion, cargar_configuracion


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


    def test_los_items_se_colorean_por_clase_solo_en_las_politicas_basadas_en_clases(self):
        from asrs.visualizacion.generador import generar_datos
        config = cargar_configuracion()
        vector_claves = config["vector_politicas_asignacion"]
        self.assertIn('basada_en_clases_replanificada', vector_claves)
        replica = generar_datos(config, [2024])["replicas"][0]
        for txt_clave, politica in zip(vector_claves, replica["politicas"]):
            txt_esperado = "clase" if txt_clave.startswith('basada_en_clases') else "sku"
            self.assertEqual(politica["color_por"], txt_esperado, txt_clave)

    def test_la_animacion_colorea_con_la_popularidad_del_escenario_generado(self):
        from asrs.politicas.zonificacion import clasificar_skus_activos
        from asrs.visualizacion.generador import _rango_por_demanda, generar_datos
        config = cargar_configuracion()
        datos = generar_datos(config, [2024, 2025])
        for replica in datos["replicas"]:
            # El mismo escenario que ven las políticas: la popularidad móvil del día 0 de esa semilla
            simulacion = Simulacion(config)
            random.seed(replica["semilla"])
            simulacion.generarSecuencias()
            vector_popularidad = simulacion.escenario_demanda.popularidad(0).astype(float)
            vector_activo = simulacion.escenario_demanda.activos(0)
            vector_clase = clasificar_skus_activos(vector_popularidad, vector_activo, config)
            vector_clase[vector_clase == 0] = vector_clase.max() # Los SKU que aún no existen se pintan como la clase de menor demanda
            self.assertEqual(replica["clase_sku"], vector_clase.tolist())
            self.assertEqual(replica["rango_sku"], _rango_por_demanda(vector_popularidad).tolist())
            self.assertAlmostEqual(sum(replica["demanda_sku"]), 1.0, places=4)
            self.assertGreater(len(replica["clase_sku"]), config["escalar_numero_skus"])
        # La plantilla lee las claves de la raíz, que son las de la primera réplica
        for txt_clave in ("clase_sku", "rango_sku", "demanda_sku"):
            self.assertEqual(datos[txt_clave], datos["replicas"][0][txt_clave])
        self.assertNotEqual(datos["replicas"][0]["clase_sku"], datos["replicas"][1]["clase_sku"])

    def test_la_animacion_con_datos_reales_conserva_la_demanda_de_la_configuracion(self):
        from asrs.config import config_a_json, config_desde_json
        from asrs.politicas.zonificacion import clasificar_skus
        from asrs.visualizacion.generador import generar_datos
        datos_config = config_a_json(cargar_configuracion())
        datos_config.update(texto_origen_tareas="datos", texto_tareas_csv=test_generacion.PruebasDatosReales.CSV, escalar_porcentaje_historial=100)
        config = config_desde_json(datos_config)
        datos = generar_datos(config, [1])
        self.assertEqual(datos["clase_sku"], clasificar_skus(config["vector_demanda_sku"], config).tolist())
        self.assertEqual(datos["replicas"][0]["clase_sku"], datos["clase_sku"])

    def test_la_animacion_graba_solo_las_primeras_tareas(self):
        from unittest import mock
        from asrs.visualizacion import generador
        config = cargar_configuracion()
        config["vector_politicas_asignacion"] = ('mas_cercana_es', 'basada_en_clases')
        # Una animación más corta que la lista: el controlador recibe los días y los retiros recortados igual que la lista
        with mock.patch.object(generador, "TAREAS_ANIMACION", 30):
            datos = generador.generar_datos(config, [2024])
        replica = datos["replicas"][0]
        self.assertEqual(len(replica["secuencia"]), 30)
        self.assertEqual(replica["tareas_totales"], TAREAS_PRUEBA)
        for politica in replica["politicas"]:
            self.assertEqual(sum(1 for e in politica["eventos"] if e[1] == 'T'), 30)


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
