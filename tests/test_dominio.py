"""Pruebas de regresión del simulador ASRS (dominio). Ejecutar con: python -m unittest discover -s tests"""
import numpy as np
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
from asrs.config import cargar_configuracion
from asrs.dominio.grua import Grua
from asrs.dominio.item import Item
from asrs.politicas.registro import crear_politica
from asrs.simulacion.simulacion import Simulacion


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
        self.assertAlmostEqual(G.vector_posicion_resposo[2], 3.0 + 2 * 1.1 + 0.5)
        self.assertTrue((E.posicionesCeldas()[:, :, :, 2] < G.vector_posicion_resposo[2]).all())
        # Al almacenar, la grúa va del E/S a la celda sin detenerse en la esquina de la estantería (0, 5)
        politica = crear_politica('mas_cercana_es', E, G, P, config)
        G.vector_eventos = []
        P.matriz_contenedor_temporal_items.append(Item(1))
        ControladorGrua(E, G, P, [], politica).almacenar((4, 6, 0))
        paradas = {(e[5], e[6]) for e in G.vector_eventos if e[1] == 'M'}
        self.assertNotIn((0.0, 5.0), paradas)


if __name__ == "__main__":
    unittest.main()
