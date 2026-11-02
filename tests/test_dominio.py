"""Pruebas de regresión del simulador ASRS (dominio). Ejecutar con: python -m unittest discover -s tests"""
import numpy as np
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
from auxiliar import _cargar_configuracion_instancia
from auxiliar import Simulacion, cargar_configuracion
from asrs.dominio.grua import Grua
from asrs.dominio.item import Item
from asrs.politicas.registro import crear_politica


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

    def test_tiempo_de_toma_y_deposito_depende_de_la_capa(self):
        # Instancia del capítulo 6: 3,43 s en el E/S y en la capa delantera, 6,86 s en la trasera
        config = cargar_configuracion()
        simulacion = Simulacion(config)
        E, G, P = simulacion.crearInstancias()
        G.entrarAlPasillo(E)
        self.assertAlmostEqual(G.tiempo_capa_delantera, 3.43)
        E.matriz_contenedor_items[3, 4, 1] = Item(1)   # capa delantera (mayor índice)
        E.matriz_contenedor_items[3, 5, 0] = Item(2)   # capa trasera
        self.assertAlmostEqual(G.cargarItemDeEstanteria((3, 4, 1)), 3.43)
        self.assertAlmostEqual(G.cargarItemDeEstanteria((3, 5, 0)), 6.86)
        self.assertAlmostEqual(G.soltarItemEnEstanteria((3, 5, 0)), 6.86)
        self.assertAlmostEqual(G.soltarItemEnEstanteria((3, 4, 1)), 3.43)
        P.matriz_contenedor_temporal_items.append(Item(3))
        G.irAPuntoES(P)
        self.assertAlmostEqual(G.cargarItemPuntoES(), 3.43)

    def test_la_instancia_por_defecto_es_la_del_capitulo_6(self):
        config = _cargar_configuracion_instancia()
        self.assertEqual(config["vector_configuracion_estanteria"].tolist(), [18, 20, 2])
        self.assertEqual(config["vector_velocidad_grua"].tolist(), [1, 1.5, 0.7])
        self.assertEqual(config["vector_aceleracion_grua"].tolist(), [0.5, 0.5, 0])
        self.assertEqual(config["escalar_numero_skus"], 180)
        self.assertEqual(config["escalar_sigma_hausman"], 0.4)
        self.assertEqual(config["escalar_largo_secuencia"], 525_600)   # un año de 365 días a 1 440 tareas por día
        self.assertEqual(config["vector_proporcion_skus_por_clase"].tolist(), [0.2, 0.2, 0.6])
        # Modelo de suministro y demanda del capítulo 6
        self.assertEqual(config["texto_modelo_demanda"], "ciclo_vida")
        self.assertEqual((config["escalar_bass_p"], config["escalar_bass_q"], config["escalar_meses_ciclo_vida"]), (0.03, 0.38, 18))
        self.assertEqual(config["vector_cuantiles_amplitud"].tolist(), [0.05, 0.10, 0.20, 0.47, 0.67])
        self.assertEqual(config["vector_probabilidades_amplitud"].tolist(), [0, 0.10, 0.50, 0.90, 1.0])
        self.assertEqual(config["vector_reparto_mes_pico"].tolist(), [0.44, 0.37, 0.19])
        self.assertEqual((config["escalar_ciclos_por_hora"], config["escalar_horas_por_dia"]), (60, 24))
        self.assertEqual((config["escalar_dias_ventana_popularidad"], config["escalar_dias_calentamiento"]), (30, 30))
        self.assertEqual(config["escalar_techo_ocupacion"], 0.95)
        self.assertEqual(config["escalar_dias_replanificacion"], 30)
        self.assertEqual(config["vector_politicas_asignacion"], ('aleatoria', 'basada_en_clases', 'basada_en_clases_replanificada', 'full_turnover',
                                                                 'full_turnover_replanificada', 'mas_cercana_es', 'mas_cercana_grua'))
        # El E/S queda 1 m sobre la base de la estantería y 1 m antes de la primera columna, con el riel en la cara frontal
        E, G, P = Simulacion(_cargar_configuracion_instancia()).crearInstancias()
        self.assertEqual(E.vector_posicion.tolist(), [-1.0, 1.0, 0])
        self.assertAlmostEqual(G.vector_posicion_resposo[2], 2 * 1.2)

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
        self.assertAlmostEqual(G.vector_posicion_resposo[2], 3.0 + 2 * config["vector_cotas_estanteria"][2] + config["escalar_separacion_riel_grua"])
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
