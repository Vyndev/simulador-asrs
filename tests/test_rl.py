"""Pruebas de regresión del simulador ASRS (rl). Ejecutar con: python -m unittest discover -s tests"""
import numpy as np
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
from asrs.config import cargar_configuracion


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
        retorno, terminado = 0.0, entorno.terminado
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

    def test_la_observacion_sigue_la_formulacion(self):
        # Estado X_t = ((l, c, d), (i, e)), tarea actual (SKU y tipo) y perfil de la demanda; acciones = posiciones disponibles
        from asrs.rl.entorno import ESCALAR_MAPAS, EntornoAsignacion
        entorno = EntornoAsignacion(self.config, 77, 2)
        observacion, _ = entorno.reset(options={"estado": 0})
        n, skus = entorno.escalar_celdas, entorno.escalar_skus
        self.assertEqual(n, 6 * 8 * 2)
        self.assertEqual(observacion.shape, (ESCALAR_MAPAS * n + 5 + 1 + 2 * skus,))
        self.assertEqual(entorno.action_space.n, n)
        self.assertTrue(np.array_equal(observacion[:n] > 0, entorno.matriz_ocupacion.reshape(-1)))          # mapa de ocupación
        self.assertTrue(np.array_equal(observacion[4 * n:5 * n] > 0, entorno.action_masks()))               # posiciones disponibles
        escalares = observacion[ESCALAR_MAPAS * n:]
        self.assertAlmostEqual(float(escalares[1] + escalares[2]), 1.0)                                     # el tipo de operación es uno solo
        self.assertAlmostEqual(float(escalares[6:6 + skus].sum()), 1.0, places=4)                           # perfil de los SKUs que entran
        self.assertAlmostEqual(float(escalares[6 + skus:].sum()), 1.0, places=4)                            # perfil de los SKUs que salen
        self.assertTrue(np.all((observacion >= 0) & (observacion <= 1)))

    def test_las_reubicaciones_tambien_son_decisiones_del_agente(self):
        from asrs.rl.entorno import ESCALAR_MAPAS, EntornoAsignacion
        entorno = EntornoAsignacion(self.config, 77, 2)
        observacion, _ = entorno.reset(options={"estado": 0})
        n, decisiones, reubicaciones = entorno.escalar_celdas, 0, 0
        while not entorno.terminado:
            reubicaciones += int(observacion[ESCALAR_MAPAS * n + 3] > 0.5)
            observacion, _, _, _, _ = entorno.step(int(np.flatnonzero(entorno.action_masks())[0]))
            decisiones += 1
        almacenes = sum(1 for actividad, _ in entorno.S if actividad == 0)
        self.assertEqual(decisiones, almacenes + reubicaciones)

    def test_reiniciar_a_medias_no_deja_hilos_colgados(self):
        import threading
        from asrs.rl.entorno import EntornoAsignacion
        entorno = EntornoAsignacion(self.config, 77, 2)
        antes = threading.active_count()
        for _ in range(4):
            entorno.reset(options={"estado": 0})
            entorno.step(int(np.flatnonzero(entorno.action_masks())[0]))
        entorno.close()
        self.assertEqual(threading.active_count(), antes)

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

    def test_la_fabrica_de_entornos_paralelos_se_puede_enviar_a_otro_proceso(self):
        import pickle
        from functools import partial
        from asrs.rl.entorno import crear_entorno_monitoreado
        fabrica = pickle.loads(pickle.dumps(partial(crear_entorno_monitoreado, self.config, 77, 2, 5)))
        entorno = fabrica()
        observacion, _ = entorno.reset()
        self.assertEqual(observacion.shape, entorno.observation_space.shape)

    def test_la_foto_de_la_red_describe_la_entrada_las_capas_y_la_decision(self):
        from asrs.rl.entrenamiento import entrenar
        from asrs.rl.entorno import EntornoAsignacion
        from asrs.rl.entrenamiento_vivo import foto_de_la_red
        entorno = EntornoAsignacion(self.config, 1000, 2)
        observacion, _ = entorno.reset(options={"estado": 0})
        for algoritmo in ("ppo", "dqn"):
            modelo, _ = entrenar(algoritmo, self.config, 150 if algoritmo == "ppo" else 1100, 2, 0, 1)
            foto = foto_de_la_red(modelo, algoritmo, observacion, entorno.action_masks(), entorno.escalar_celdas, 150)
            self.assertEqual(len(foto["entrada"]), 6)
            self.assertEqual(len(foto["entrada"][0]), entorno.escalar_celdas)
            self.assertEqual(len(foto["activaciones"]), 2)
            self.assertEqual(len(foto["pesos"]), 3)
            self.assertEqual(len(foto["salida"]["valores"]), entorno.escalar_celdas)
            self.assertTrue(entorno.action_masks()[foto["salida"]["elegida"]])          # la decisión nunca es un canal sin espacio
            if algoritmo == "ppo":
                self.assertAlmostEqual(sum(v for v in foto["salida"]["valores"] if v is not None), 1.0, places=2)

    def test_el_gestor_ordena_los_eventos_del_proceso(self):
        from asrs.visualizacion.gestor_entrenamiento import Entrenamiento
        entrenamiento = object.__new__(Entrenamiento)
        entrenamiento.estado, entrenamiento.metricas, entrenamiento.fotos, entrenamiento.secuencia = "iniciando", [], [], 0
        entrenamiento.referencias, entrenamiento.evaluacion, entrenamiento.cerrojo = {}, {}, __import__("threading").Lock()
        entrenamiento.id, entrenamiento.algoritmo, entrenamiento.inicio, entrenamiento.fin, entrenamiento.error = "x", "ppo", None, None, None
        for evento in ({"tipo": "inicio", "pasos": 10}, {"tipo": "referencia", "clave": "aleatoria", "media": 5.0}, {"tipo": "metrica", "pasos": 1},
                       {"tipo": "snapshot", "pasos": 1}, {"tipo": "evaluacion", "clave": "PPO", "media": 4.0}, {"tipo": "fin", "segundos": 1}):
            entrenamiento._aplicar(evento)
        self.assertEqual(entrenamiento.estado, "listo")
        self.assertEqual(entrenamiento.publico(0)["secuencia"], 2)
        self.assertEqual(len(entrenamiento.publico(1)["fotos"]), 1)                    # solo lo posterior a la secuencia indicada
        self.assertEqual(entrenamiento.publico(2)["metricas"], [])

    def test_las_clasicas_parten_del_mismo_estado(self):
        from asrs.rl.evaluacion import evaluar_politica_clasica
        a = evaluar_politica_clasica(self.config, "mas_cercana_es", 77, [0, 1])["tiempos"]
        b = evaluar_politica_clasica(self.config, "mas_cercana_es", 77, [0, 1])["tiempos"]
        self.assertTrue(np.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
