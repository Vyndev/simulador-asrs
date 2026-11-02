"""Pruebas de regresión del simulador ASRS (politicas). Ejecutar con: python -m unittest discover -s tests"""
import numpy as np
import random
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
from auxiliar import Simulacion, cargar_configuracion
from asrs.dominio.item import Item
from asrs.politicas.recuperacion import PoliticaRecuperacionFCFS, PoliticaRecuperacionFCFSAccesible
from asrs.politicas.registro import POLITICAS_ASIGNACION, crear_politica
from asrs.politicas.restriccion import MatricesRestriccion
from asrs.politicas.zonificacion import agrupar_skus_en_clases, construir_zonificacion


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
        zonas = construir_zonificacion(tiempos, [0.75, 0.25], [2, 1])
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

    def test_minimizacion_de_varianza_llena_todos_los_fondos_antes_que_los_frentes(self):
        # Lerher et al. (2010) y Lehmann y Hußmann (2023): hasta el 50 % de ocupación solo se usa la capa más profunda,
        # así no hay reubicaciones; el frente de un canal no se ocupa mientras quede un fondo libre en otro canal
        for clave in ('aleatoria', 'mas_cercana_es', 'mas_cercana_grua'):
            simulacion = Simulacion()
            E, G, P = simulacion.crearInstancias()
            politica = crear_politica(clave, E, G, P, simulacion.config)
            escalar_canales = E.matriz_contenedor_items.shape[0] * E.matriz_contenedor_items.shape[1]
            for _ in range(escalar_canales):
                celda = politica.asignar(1)
                self.assertEqual(celda[2], 0, clave)
                E.matriz_contenedor_items[celda] = Item(1)
            # Con todos los fondos ocupados, el siguiente va a un frente
            self.assertEqual(politica.asignar(1)[2], 1, clave)

    def test_mas_cercana_al_es_elige_el_canal_mas_cercano(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        politica = crear_politica('mas_cercana_es', E, G, P, simulacion.config)
        # Varias filas de la primera columna empatan en tiempo desde el E/S (domina el eje horizontal): se elige una de ellas, en el fondo
        celda = politica.asignar(1)
        matriz_criterio = politica.matrizCriterio()
        self.assertEqual(matriz_criterio[celda], matriz_criterio.min())
        self.assertEqual(celda[1:], (0, 0))

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

    def test_las_zonas_se_dimensionan_con_el_inventario(self):
        from asrs.generacion.probabilidades import perfil_inventario
        from asrs.utilidades import repartir_entero
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        demanda = np.array([0.64, 0.16, 0.16, 0.04])
        canales = E.matriz_contenedor_items.shape[0] * E.matriz_contenedor_items.shape[1]
        for txt_perfil in ("eoq", "demanda"):
            config = dict(simulacion.config, vector_demanda_sku=demanda, escalar_numero_skus=4, texto_perfil_inventario=txt_perfil)
            politica = crear_politica('full_turnover', E, G, P, config)
            canales_por_sku = [int((politica.matriz_zonificacion[:, :, 0] == sku).sum()) for sku in range(1, 5)]
            self.assertEqual(canales_por_sku, list(repartir_entero(canales, perfil_inventario(demanda, txt_perfil), escalar_minimo=1)), txt_perfil)

    def test_basada_en_clases_agrupa_los_skus_ella_misma(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        config = dict(simulacion.config, vector_demanda_sku=np.array([0.05, 0.4, 0.1, 0.3, 0.15]), escalar_numero_skus=5,
                      texto_criterio_clases="skus", vector_proporcion_skus_por_clase=np.array([0.4, 0.6]))
        politica = crear_politica('basada_en_clases', E, G, P, config)
        self.assertEqual([politica.etiquetaZona(sku) for sku in range(1, 6)], [2, 1, 2, 1, 2])
        self.assertEqual(politica.matriz_zonificacion[0, 0, 0], 1)           # la clase de los SKUs más demandados queda más cerca


class PruebasClasesPorDemanda(unittest.TestCase):
    def test_partes_iguales_reparte_la_demanda_sin_indicar_cuantos_skus(self):
        from asrs.politicas.zonificacion import agrupar_skus_por_demanda
        from asrs.generacion.probabilidades import demanda_hausman
        demanda = demanda_hausman(30, 0.5)
        clase = agrupar_skus_por_demanda(demanda, [1, 1, 1])
        self.assertEqual(sorted(set(clase)), [1, 2, 3])
        for k in (1, 2, 3):
            self.assertAlmostEqual(float(demanda[clase == k].sum()), 1 / 3, delta=0.06)          # cada clase queda cerca de un tercio de la demanda
        self.assertTrue(np.all(np.diff(clase[np.argsort(-demanda, kind="stable")]) >= 0))        # la clase 1 son los más demandados
        self.assertLess(int((clase == 1).sum()), int((clase == 3).sum()))                         # pocos SKUs muy demandados y muchos poco demandados

    def test_pareto_da_80_15_5_de_la_demanda(self):
        from asrs.politicas.zonificacion import clasificar_skus
        from asrs.generacion.probabilidades import demanda_hausman
        demanda = demanda_hausman(30, 0.5)
        clase = clasificar_skus(demanda, {"texto_criterio_clases": "pareto"})
        self.assertEqual([round(float(demanda[clase == k].sum()), 2) for k in (1, 2, 3)], [0.80, 0.15, 0.05])

    def test_cada_clase_tiene_al_menos_un_sku_incluso_con_demanda_muy_concentrada(self):
        from asrs.politicas.zonificacion import agrupar_skus_por_demanda
        demanda = np.array([0.97, 0.01, 0.01, 0.005, 0.005])
        clase = agrupar_skus_por_demanda(demanda, [1, 1, 1, 1])
        self.assertEqual(sorted(set(clase)), [1, 2, 3, 4])

    def test_la_politica_usa_el_criterio_de_la_configuracion(self):
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        por_defecto = crear_politica('basada_en_clases', E, G, P, simulacion.config)
        self.assertEqual(simulacion.config["texto_criterio_clases"], "skus")
        self.assertEqual(int(por_defecto.vector_clase_de_sku.max()), 3)
        # Con el criterio de partes iguales de la demanda, el número de clases es el configurado
        cinco = crear_politica('basada_en_clases', E, G, P, dict(simulacion.config, texto_criterio_clases="iguales", escalar_numero_clases=5))
        self.assertEqual(int(cinco.vector_clase_de_sku.max()), 5)


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

    def test_fcfs_accesible_prefiere_el_mas_antiguo_sin_nada_delante(self):
        from asrs.politicas.recuperacion import PoliticaRecuperacionFCFSAccesible
        simulacion = Simulacion()
        E, _, _ = simulacion.crearInstancias()
        E.matriz_contenedor_items[3, 3, 0] = Item(1)  # el más antiguo, al fondo
        E.matriz_contenedor_items[3, 3, 1] = Item(1)  # más nuevo, delante
        E.matriz_contenedor_items[5, 5, 0] = Item(1)  # el más nuevo, al fondo pero sin nada delante
        self.assertEqual(PoliticaRecuperacionFCFS().seleccionar(E, 1), (3, 3, 0))
        self.assertEqual(PoliticaRecuperacionFCFSAccesible().seleccionar(E, 1), (3, 3, 1))
        # si ninguna unidad del SKU está accesible, se retira la más antigua (y habrá que reubicar)
        E.matriz_contenedor_items[6, 6, 0] = Item(2)
        E.matriz_contenedor_items[6, 6, 1] = Item(3)
        self.assertEqual(PoliticaRecuperacionFCFSAccesible().seleccionar(E, 2), (6, 6, 0))

    def test_tras_reubicar_va_directo_a_la_celda_objetivo(self):
        from asrs.control.controlador_grua import ControladorGrua
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        objetivo, bloquea = (4, 6, 0), (4, 6, 1)
        E.matriz_contenedor_items[objetivo] = Item(1)
        E.matriz_contenedor_items[bloquea] = Item(2)
        random.seed(1)
        G.vector_eventos = []
        ControladorGrua(E, G, P, [(1, 1)], crear_politica('aleatoria', E, G, P, simulacion.config)).ejecutarSecuencia()

        eventos = G.vector_eventos
        indice_dejar = next(i for i, e in enumerate(eventos) if e[1] == 'D')          # la grúa deja el ítem que estorbaba
        indice_retirar = next(i for i, e in enumerate(eventos) if e[1] == 'P' and tuple(e[2:5]) == objetivo)
        destino_reubicacion = E.posicionesCeldas()[tuple(eventos[indice_dejar][2:5])][:2]
        self.assertTrue(np.any(destino_reubicacion != E.vector_posicion[:2]))     # lejos de la esquina, así volver a ella costaría tiempo
        movimientos = [e for e in eventos[indice_dejar + 1:indice_retirar] if e[1] == 'M' and e[2] > 0]
        # Sale del canal y va de la celda de reubicación a la celda objetivo (xy y luego z hasta el fondo), sin pasar por la esquina
        self.assertEqual(len(movimientos), 2)
        self.assertEqual(movimientos[1][3:5], [round(float(x), 4) for x in destino_reubicacion])
        self.assertEqual(movimientos[1][5:7], [round(float(x), 4) for x in E.posicionesCeldas()[objetivo][:2]])
        self.assertIsInstance(E.matriz_contenedor_items[objetivo], int)
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


class PruebasControladorDinamico(unittest.TestCase):
    def test_controlador_avisa_el_inicio_de_cada_dia(self):
        from asrs.control.controlador_grua import ControladorGrua
        from asrs.politicas.mas_cercana_es import PoliticaMasCercanaAlES

        class PoliticaConDias(PoliticaMasCercanaAlES):
            def __init__(self, *args):
                super().__init__(*args)
                self.vector_dias = []

            def iniciarDia(self, escalar_dia):
                self.vector_dias.append(escalar_dia)

        simulacion = Simulacion()
        vector_secuencia = [(0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2)]
        for vector_dia_de_tarea, vector_esperado in (([0, 0, 1, 1, 1, 2], [0, 1, 2]), (None, [])):
            E, G, P = simulacion.crearInstancias()
            politica = PoliticaConDias(E, G, P, simulacion.config)
            ControladorGrua(E, G, P, vector_secuencia, politica, vector_dia_de_tarea=vector_dia_de_tarea).ejecutarSecuencia()
            self.assertEqual(politica.vector_dias, vector_esperado)

    def test_retiros_se_cuentan_aparte(self):
        from asrs.control.controlador_grua import ControladorGrua
        simulacion = Simulacion()
        E, G, P = simulacion.crearInstancias()
        politica = crear_politica('mas_cercana_es', E, G, P, simulacion.config)
        C = ControladorGrua(E, G, P, [(0, 1), (1, 1)], politica, conjunto_retiros={1})
        C.ejecutarSecuencia()
        M = C.instancia_metricas
        vector_fila = M.recuperarInformcion()
        # [..., I0, I1, I2, R0, R1, O0, O1]
        self.assertEqual(vector_fila[-4], 1)
        self.assertGreater(vector_fila[-3], 0)
        self.assertAlmostEqual(vector_fila[-3], M.tiempo_total_recuperacion)
        self.assertEqual(vector_fila[3], 1)
        self.assertEqual(len(vector_fila), 31)

    def test_zona_llena_se_relaja_al_almacenar(self):
        from asrs.control.controlador_grua import ControladorGrua
        config = dict(cargar_configuracion(), vector_configuracion_estanteria=np.array([2, 3, 2]),
                      vector_demanda_sku=np.array([0.2, 0.3, 0.5]), escalar_numero_skus=3)
        simulacion = Simulacion(config)
        E, G, P = simulacion.crearInstancias()
        politica = crear_politica('full_turnover', E, G, P, config)
        matriz_zona_1 = politica.matriz_zonificacion == politica.etiquetaZona(1)
        for indice in zip(*np.nonzero(matriz_zona_1)):
            E.matriz_contenedor_items[indice] = Item(1)
        C = ControladorGrua(E, G, P, [(0, 1), (0, 2)], politica)
        C.ejecutarSecuencia()
        # El almacenamiento del SKU 1 no se descarta aunque su zona esté llena: queda fuera de ella
        self.assertEqual(C.instancia_metricas.cantidad_almacenamientos, 2)
        vector_indices_1 = [i for i in np.ndindex(E.matriz_contenedor_items.shape)
                            if isinstance(E.matriz_contenedor_items[i], Item) and E.matriz_contenedor_items[i].escalar_sku == 1]
        self.assertEqual(sum(1 for i in vector_indices_1 if not matriz_zona_1[i]), 1)
        # El SKU 2 tiene espacio en su zona y se queda en ella
        vector_indices_2 = [i for i in np.ndindex(E.matriz_contenedor_items.shape)
                            if isinstance(E.matriz_contenedor_items[i], Item) and E.matriz_contenedor_items[i].escalar_sku == 2]
        self.assertEqual(len(vector_indices_2), 1)
        self.assertEqual(politica.matriz_zonificacion[vector_indices_2[0]], politica.etiquetaZona(2))


if __name__ == "__main__":
    unittest.main()
