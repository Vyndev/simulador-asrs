"""Pruebas de regresión del simulador ASRS (visualizacion). Ejecutar con: python -m unittest discover -s tests"""
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
from asrs.config import cargar_configuracion


class PruebasVisualizacion(unittest.TestCase):
    def test_reloj_grabado_coincide_con_las_metricas(self):
        from asrs.visualizacion.generador import generar_datos
        datos = generar_datos(cargar_configuracion(), [2024])
        replica = datos["replicas"][0]
        self.assertEqual(len(replica["politicas"]), 10)
        for politica in replica["politicas"]:
            self.assertAlmostEqual(politica["reloj_final"], politica["tiempo_total"], places=2, msg=politica["nombre"])
            self.assertEqual(politica["eventos"][-1][1], 'F')
            self.assertEqual(sum(1 for e in politica["eventos"] if e[1] == 'T'), len(replica["secuencia"]))


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
            self.assertEqual(len(resultado["estadistica"]["politicas"]), 4)
        finally:
            servidor.shutdown()
            servidor.server_close()


if __name__ == "__main__":
    unittest.main()
