"""Pruebas del perfil de SKU y de las celdas candidatas del agente. Ejecutar con: python -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asrs.agente.candidatos import celdas_candidatas, mascara_candidatas
from asrs.agente.perfil import EPSILON_TASA, historia_en_dias, perfil_de_skus
from asrs.config import cargar_configuracion
from asrs.dominio.item import Item
from asrs.generacion.reposicion import SIN_HISTORIA_PREVIA, generar_escenario
from asrs.simulacion.simulacion import Simulacion


class EscenarioDeMano:
    # Escenario mínimo: 3 SKU y 70 días de registro, de los cuales los 30 primeros son la prehistoria
    def __init__(self, matriz_recuperaciones, vector_primera_llegada, vector_lanzamiento=None):
        self.matriz_recuperaciones_dia = np.asarray(matriz_recuperaciones, dtype=np.int64)
        self.vector_dia_primera_llegada = np.asarray(vector_primera_llegada, dtype=float)
        self.vector_dia_lanzamiento = np.zeros(self.matriz_recuperaciones_dia.shape[1]) if vector_lanzamiento is None else np.asarray(vector_lanzamiento, dtype=float)
        self.escalar_dias_prehistoria = 30

    def recuperaciones(self, escalar_desde, escalar_hasta):
        escalar_desde = min(max(int(escalar_desde) + 30, 0), len(self.matriz_recuperaciones_dia))
        escalar_hasta = min(max(int(escalar_hasta) + 30, 0), len(self.matriz_recuperaciones_dia))
        return self.matriz_recuperaciones_dia[escalar_desde:max(escalar_desde, escalar_hasta)].copy()


def escenario_ejemplo():
    # SKU 1 maduro: 2 por día siempre. SKU 2 nuevo: llegó el día 10 de la operación y vende 1 por día desde entonces. SKU 3 sin ventas
    matriz = np.zeros((70, 3), dtype=int)
    matriz[:, 0] = 2
    matriz[40:, 1] = 1
    return EscenarioDeMano(matriz, [SIN_HISTORIA_PREVIA, 10, np.inf])


class PruebasPerfil(unittest.TestCase):
    def test_historia_acotada_a_la_ventana(self):
        escenario = escenario_ejemplo()
        self.assertEqual(historia_en_dias(escenario, 10, 30).tolist(), [30.0, 0.0, 0.0])    # el SKU 2 llega el día 10
        self.assertEqual(historia_en_dias(escenario, 13, 30).tolist(), [30.0, 3.0, 0.0])
        self.assertEqual(historia_en_dias(escenario, 60, 30).tolist(), [30.0, 30.0, 0.0])   # acotada a 30

    def test_sku_maduro_estable(self):
        perfil = perfil_de_skus(escenario_ejemplo(), 20)
        self.assertAlmostEqual(perfil[0, 0], np.log1p(2.0))    # tasa 2 por día
        self.assertAlmostEqual(perfil[0, 1], 0.0)              # sin tendencia
        self.assertAlmostEqual(perfil[0, 2], 1.0)              # historia completa

    def test_sku_nuevo_se_mide_con_la_historia_que_tiene(self):
        # Con 3 días de historia y 3 recuperaciones, la tasa es 1 por día y no 3 / 30
        perfil = perfil_de_skus(escenario_ejemplo(), 13)
        self.assertAlmostEqual(perfil[1, 0], np.log1p(3 / 3))
        self.assertAlmostEqual(perfil[1, 1], 0.0)              # aún no hay una mitad previa
        self.assertAlmostEqual(perfil[1, 2], 3 / 30)

    def test_sku_sin_ventas_tiene_nivel_cero(self):
        perfil = perfil_de_skus(escenario_ejemplo(), 20)
        self.assertEqual(perfil[2].tolist(), [0.0, 0.0, 0.0])

    def test_tendencia_positiva_y_negativa(self):
        # Un SKU que vende 1 por día y luego 3 por día (sube) y otro que pasa de 3 a 1 (baja), con historia completa
        matriz = np.zeros((70, 2), dtype=int)
        matriz[:55, 0], matriz[55:, 0] = 1, 3
        matriz[:55, 1], matriz[55:, 1] = 3, 1
        perfil = perfil_de_skus(EscenarioDeMano(matriz, [SIN_HISTORIA_PREVIA] * 2), 40)     # días 25 a 39 contra 10 a 24
        self.assertAlmostEqual(perfil[0, 1], np.log((3 + EPSILON_TASA) / (1 + EPSILON_TASA)))
        self.assertAlmostEqual(perfil[1, 1], np.log((1 + EPSILON_TASA) / (3 + EPSILON_TASA)))
        self.assertGreater(perfil[0, 1], 0)
        self.assertLess(perfil[1, 1], 0)

    def test_no_mira_hacia_adelante(self):
        # Cambiar lo que pasa desde el día dado no cambia el perfil de ese día
        escenario = escenario_ejemplo()
        perfil_antes = perfil_de_skus(escenario, 25)
        escenario.matriz_recuperaciones_dia[30 + 25:, :] = 99
        self.assertTrue(np.array_equal(perfil_de_skus(escenario, 25), perfil_antes))

    def test_edad_completa_es_opcional(self):
        escenario = escenario_ejemplo()
        escenario.vector_dia_lanzamiento = np.array([-100.0, 5.0, 0.0])
        self.assertEqual(perfil_de_skus(escenario, 20).shape[1], 3)
        perfil = perfil_de_skus(escenario, 20, escalar_duracion_ciclo=500)
        self.assertEqual(perfil.shape[1], 4)
        self.assertAlmostEqual(perfil[0, 3], 120 / 500)
        self.assertAlmostEqual(perfil[1, 3], 15 / 500)

    def test_con_un_escenario_generado(self):
        config = dict(cargar_configuracion(), escalar_largo_secuencia=40 * 1440)
        escenario = generar_escenario(config, np.random.default_rng(2))
        escalar_dia = 35
        perfil = perfil_de_skus(escenario, escalar_dia)
        vector_activo = escenario.activos(escalar_dia)
        self.assertEqual(perfil.shape, (len(escenario.vector_inventario_inicial), 3))
        self.assertTrue(np.all(np.isfinite(perfil)))
        self.assertTrue(np.all((perfil[:, 2] >= 0) & (perfil[:, 2] <= 1)))
        # El nivel de los SKU establecidos es la tasa de la popularidad móvil: ln(1 + popularidad / 30)
        vector_establecido = vector_activo & (perfil[:, 2] == 1.0)
        self.assertGreater(int(vector_establecido.sum()), 100)
        self.assertTrue(np.allclose(perfil[vector_establecido, 0], np.log1p(escenario.popularidad(escalar_dia)[vector_establecido] / 30)))


class PruebasEscenarioObservable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(cargar_configuracion(), escalar_largo_secuencia=40 * 1440)
        cls.escenario = generar_escenario(cls.config, np.random.default_rng(4))

    def test_la_popularidad_es_la_suma_de_las_recuperaciones_pasadas(self):
        for escalar_dia in (0, 10, 38):
            vector_activo = self.escenario.activos(escalar_dia)
            vector_suma = self.escenario.recuperaciones(escalar_dia - 30, escalar_dia).sum(axis=0)
            self.assertTrue(np.array_equal(vector_suma[vector_activo], self.escenario.popularidad(escalar_dia)[vector_activo]))

    def test_las_recuperaciones_diarias_coinciden_con_la_lista(self):
        escenario = self.escenario
        vector_cuenta = np.zeros_like(escenario.matriz_recuperaciones_dia[30:])
        for escalar_indice, (escalar_actividad, escalar_sku) in enumerate(escenario.lista_operacion):
            if escalar_actividad == 1 and escalar_indice not in escenario.conjunto_retiros:
                vector_cuenta[escenario.vector_dia_de_tarea[escalar_indice], escalar_sku - 1] += 1
        escalar_dias = len(vector_cuenta)
        self.assertTrue(np.array_equal(escenario.recuperaciones(0, escalar_dias), vector_cuenta[:escalar_dias]))

    def test_primera_llegada(self):
        escenario = self.escenario
        vector_primera = escenario.vector_dia_primera_llegada
        # Los que tenían existencias el día 0 de la prehistoria tienen historia completa
        self.assertEqual(int((vector_primera == SIN_HISTORIA_PREVIA).sum()), 180)
        # Un SKU nacido en la operación tiene como primera llegada su primer almacenamiento de la lista, y uno nacido en la prehistoria, un día negativo
        vector_primer_almacenamiento = {}
        for escalar_indice, (escalar_actividad, escalar_sku) in enumerate(escenario.lista_operacion):
            if escalar_actividad == 0:
                vector_primer_almacenamiento.setdefault(escalar_sku, escenario.vector_dia_de_tarea[escalar_indice])
        escalar_comprobados = 0
        for escalar_sku, escalar_dia in vector_primer_almacenamiento.items():
            if vector_primera[escalar_sku - 1] >= 0:    # los nacidos en la prehistoria tienen su primera llegada antes del día 0
                self.assertEqual(vector_primera[escalar_sku - 1], escalar_dia)
                escalar_comprobados += 1
        self.assertGreater(escalar_comprobados, 5)
        self.assertTrue(np.any((vector_primera < 0) & (vector_primera > SIN_HISTORIA_PREVIA)))


def estanteria_con(unidades, filas=3, columnas=2, capas=2):
    # Estantería pequeña con las unidades dadas, un diccionario {(fila, columna, capa): sku}
    config = dict(cargar_configuracion(), vector_configuracion_estanteria=np.array([filas, columnas, capas]))
    E, G, P = Simulacion(config).crearInstancias()
    for celda, escalar_sku in unidades.items():
        E.matriz_contenedor_items[celda] = Item(escalar_sku)
    return E, config


class PruebasCandidatas(unittest.TestCase):
    def tiempos(self, E):
        return np.arange(E.matriz_contenedor_items.size, dtype=float).reshape(E.matriz_contenedor_items.shape)

    def test_estanteria_vacia_ofrece_solo_los_fondos(self):
        # Regla de minimización de varianza: con fondos libres no se ofrece ningún frente
        E, _ = estanteria_con({})
        matriz = mascara_candidatas(E)
        self.assertEqual(int(matriz.sum()), 6)
        self.assertTrue(matriz[:, :, 0].all() and not matriz[:, :, 1].any())

    def test_frentes_solo_con_todos_los_fondos_ocupados(self):
        E, _ = estanteria_con({(0, 0, 0): 5})
        self.assertEqual(int(mascara_candidatas(E)[:, :, 1].sum()), 0)        # quedan 5 fondos libres
        E, _ = estanteria_con({(f, c, 0): 5 for f in range(3) for c in range(2)})
        matriz = mascara_candidatas(E)
        self.assertEqual(int(matriz.sum()), 6)
        self.assertTrue(matriz[:, :, 1].all())

    def test_canal_con_el_frente_ocupado_no_ofrece_nada(self):
        E, _ = estanteria_con({(1, 1, 1): 7})        # el frente ocupado bloquea el fondo libre
        self.assertFalse(mascara_candidatas(E)[1, 1].any())

    def test_la_regla_se_aplica_despues_de_la_restriccion(self):
        # Al reubicar se excluye el canal que se despeja; si es el único con fondo libre, se ofrecen los frentes
        E, _ = estanteria_con({(f, c, 0): 5 for f in range(3) for c in range(2) if (f, c) != (2, 1)})
        matriz_extra = np.ones(E.matriz_contenedor_items.shape, dtype=bool)
        matriz_extra[2, 1, :] = False
        self.assertTrue(mascara_candidatas(E)[2, 1, 0] and int(mascara_candidatas(E).sum()) == 1)
        matriz = mascara_candidatas(E, matriz_extra)
        self.assertEqual(int(matriz.sum()), 5)
        self.assertFalse(matriz[:, :, 0].any() or matriz[2, 1].any())

    def test_caracteristicas(self):
        E, _ = estanteria_con({(f, c, 0): 1 + (f + c) % 2 for f in range(3) for c in range(2)})   # fondos llenos con los SKU 1 y 2
        matriz_perfil = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])        # SKU 1 y SKU 2
        matriz_indices, matriz_caracteristicas = celdas_candidatas(E, self.tiempos(E), matriz_perfil)
        self.assertEqual(matriz_caracteristicas.shape, (6, 5))
        fila = [i for i, t in enumerate(map(tuple, matriz_indices)) if t == (0, 1, 1)][0]
        self.assertEqual(matriz_caracteristicas[fila, 1], 1.0)                                       # capa delantera
        self.assertEqual(matriz_caracteristicas[fila, 2:].tolist(), [0.4, 0.5, 0.6])                 # perfil del SKU 2, el de atrás
        self.assertTrue(np.all((matriz_caracteristicas[:, 0] >= 0) & (matriz_caracteristicas[:, 0] <= 1)))
        E, _ = estanteria_con({})
        matriz_indices, matriz_caracteristicas = celdas_candidatas(E, self.tiempos(E), matriz_perfil)
        self.assertTrue(np.all(matriz_caracteristicas[:, 1:] == 0))                                 # fondos: capa 0 y nadie atrás

    def test_estanteria_llena_no_tiene_candidatas(self):
        E, _ = estanteria_con({(f, c, k): 1 for f in range(3) for c in range(2) for k in range(2)})
        matriz_indices, matriz_caracteristicas = celdas_candidatas(E, self.tiempos(E), np.zeros((1, 3)))
        self.assertEqual(len(matriz_indices), 0)
        self.assertEqual(matriz_caracteristicas.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
