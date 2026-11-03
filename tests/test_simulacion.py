"""Pruebas de regresión del simulador ASRS (simulacion). Ejecutar con: python -m unittest discover -s tests"""
import numpy as np
import os
import unittest

import auxiliar  # noqa: F401  (deja la raíz del proyecto en sys.path)
import test_generacion
from auxiliar import Simulacion, cargar_configuracion, preparar_estado
from asrs.dominio.item import Item
from asrs.politicas.registro import POLITICAS_ASIGNACION
from asrs.politicas.restriccion import MatricesRestriccion


class PruebasResultados(unittest.TestCase):
    def test_encabezado_de_resultados(self):
        import csv
        import tempfile
        from asrs.simulacion.resultados import guardar_resultados_csv
        with tempfile.TemporaryDirectory() as txt_carpeta:
            txt_ruta = os.path.join(txt_carpeta, 'resultados.csv')
            guardar_resultados_csv([], txt_ruta)
            with open(txt_ruta, newline='') as archivo:
                vector_encabezado = next(csv.reader(archivo, delimiter=';'))
        self.assertEqual(len(vector_encabezado), 33)
        self.assertEqual(vector_encabezado[-4:], ["Cant_RetirosFinCiclo", "TiempoTotal_RetirosFinCiclo", "Cant_Operaciones", "TiempoTotal_Operacion"])


class PruebasEscenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulacion = Simulacion()
        cls.SA, cls.S = cls.simulacion.generarSecuencias()

    def test_largo_de_secuencias(self):
        self.assertEqual(len(self.S), self.simulacion.config["escalar_largo_secuencia"])

    def test_llenado_reproduce_el_inventario_del_calentamiento(self):
        for clave in POLITICAS_ASIGNACION:
            simulacion, E, _, _, _ = preparar_estado(clave)
            # Con la regla de zona llena ninguna política descarta unidades, ni siquiera la de zonas dedicadas
            self.assertEqual(np.count_nonzero(E.matriz_contenedor_items), len(simulacion.escenario_demanda.lista_llenado), clave)

    def test_politicas_con_zonas_respetan_su_zona(self):
        for clave in ('basada_en_clases', 'full_turnover'):
            _, E, _, _, politica = preparar_estado(clave)
            matriz_ocupacion = MatricesRestriccion().ocupacion(E)
            matriz_libres_accesibles = np.logical_not(matriz_ocupacion) & MatricesRestriccion().accesibilidad(matriz_ocupacion)
            for indice in np.ndindex(E.matriz_contenedor_items.shape):
                item = E.matriz_contenedor_items[indice]
                if isinstance(item, Item) and politica.matriz_zonificacion[indice] != politica.etiquetaZona(item.escalar_sku):
                    # Regla de zona llena: una unidad fuera de su zona solo es posible si la zona no tiene celdas libres accesibles
                    # Durante el llenado no hay recuperaciones, así que una zona que se llenó sigue llena
                    matriz_zona = politica.matriz_zonificacion == politica.etiquetaZona(item.escalar_sku)
                    self.assertFalse((matriz_zona & matriz_libres_accesibles).any(), (clave, indice))

    def test_no_quedan_fondos_libres_tras_un_item_frontal(self):
        for clave in POLITICAS_ASIGNACION:
            simulacion, E, G, P, politica = preparar_estado(clave)
            from asrs.control.controlador_grua import ControladorGrua
            ControladorGrua(E, G, P, self.S, politica).ejecutarSecuencia()
            ocupacion = MatricesRestriccion().ocupacion(E)
            self.assertEqual(int((~ocupacion[:, :, 0] & ocupacion[:, :, 1]).sum()), 0, clave)

    def test_replica_reproducible_y_completa(self):
        config = cargar_configuracion()
        config["vector_politicas_recuperacion"] = ('fcfs', 'fcfs_accesible')
        r1 = Simulacion.cargarReplica((config, 7))
        r2 = Simulacion.cargarReplica((config, 7))
        self.assertEqual(r1, r2)
        self.assertEqual(len(r1), len(config["vector_politicas_asignacion"]) * len(config["vector_politicas_recuperacion"]))
        self.assertEqual(len(r1[0]), 33)
        self.assertEqual([fila[0] for fila in r1[::2]],
                         ['Aleatoria', 'BasadaEnClases', 'FullTurnover', 'MasCercanaAlES', 'MasCercanaALaGrua'])
        self.assertEqual({(fila[1]) for fila in r1}, {'FCFS', 'FCFS-Accesible'})

    def test_resumen_de_escenario_por_replica(self):
        import csv
        import tempfile
        from asrs.simulacion.resultados import guardar_resumen_escenarios_csv
        config = cargar_configuracion()
        filas, dic_resumen = Simulacion.cargarReplicaConEscenario((config, 7))
        self.assertEqual(filas, Simulacion.cargarReplica((config, 7)))
        for txt_clave in ("semilla", "ocupacion_media", "demanda_no_atendida"):
            self.assertIn(txt_clave, dic_resumen)
        self.assertEqual(dic_resumen["semilla"], 7)
        with tempfile.TemporaryDirectory() as txt_carpeta:
            txt_ruta = os.path.join(txt_carpeta, 'escenario.csv')
            guardar_resumen_escenarios_csv([dic_resumen], txt_ruta)
            with open(txt_ruta, newline='') as archivo:
                vector_filas = list(csv.reader(archivo, delimiter=';'))
        self.assertEqual(vector_filas[0], list(dic_resumen))
        self.assertEqual(len(vector_filas), 2)
        self.assertEqual(vector_filas[1][0], "7")

    def test_fcfs_entre_accesibles_reubica_menos(self):
        # Con la misma asignación y el mismo azar, elegir entre los accesibles nunca debería reubicar más que FCFS estricto
        config = cargar_configuracion()
        config["vector_politicas_recuperacion"] = ('fcfs', 'fcfs_accesible')
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
        self.assertEqual(copia["texto_modelo_demanda"], original["texto_modelo_demanda"])
        self.assertEqual(copia["escalar_meses_ciclo_vida"], original["escalar_meses_ciclo_vida"])
        self.assertEqual(copia["vector_cuantiles_amplitud"].tolist(), original["vector_cuantiles_amplitud"].tolist())

    def test_la_configuracion_por_defecto_pasa_su_propia_validacion(self):
        from asrs.config import cargar_configuracion as cargar_defecto, config_a_json, config_desde_json
        # El valor por defecto (un año de tareas) supera el límite de los archivos reales, pero no el de las listas generadas
        copia = config_desde_json(config_a_json(cargar_defecto()))
        self.assertEqual(copia["escalar_largo_secuencia"], 525_600)

    def test_limite_de_tareas_generadas(self):
        from asrs.config import LIMITE_TAREAS_GENERADAS, config_desde_json
        self.assertEqual(LIMITE_TAREAS_GENERADAS, 600_000)
        self.assertEqual(config_desde_json({"escalar_largo_secuencia": 600_000})["escalar_largo_secuencia"], 600_000)
        with self.assertRaises(ValueError):
            config_desde_json({"escalar_largo_secuencia": 600_001})

    def test_techo_de_ocupacion_sin_clave_se_sube_para_ocupaciones_altas(self):
        from asrs.config import config_desde_json
        # La herramienta web no muestra el techo: con una ocupación de 95 % o más y sin techo explícito, el techo pasa a 1
        for escalar_ocupacion in (0.95, 0.97):
            config = config_desde_json({"escalar_nivel_ocupacion": escalar_ocupacion})
            self.assertEqual(config["escalar_techo_ocupacion"], 1.0)
        self.assertEqual(config_desde_json({"escalar_nivel_ocupacion": 0.9})["escalar_techo_ocupacion"], 0.95)

    def test_techo_de_ocupacion_no_aplica_con_datos_reales(self):
        from asrs.config import config_a_json, config_desde_json
        datos = config_a_json(cargar_configuracion())
        datos.update(texto_origen_tareas="datos", texto_tareas_csv=test_generacion.PruebasDatosReales.CSV, escalar_nivel_ocupacion=1.0)
        self.assertEqual(config_desde_json(datos)["escalar_nivel_ocupacion"], 1.0)

    def test_techo_explicito_que_no_supera_la_ocupacion_se_rechaza_con_mensaje_claro(self):
        from asrs.config import config_desde_json
        for caso in ({"escalar_nivel_ocupacion": 0.95, "escalar_techo_ocupacion": 0.95},
                     {"escalar_nivel_ocupacion": 1.0}):    # sin techo explícito, pero el techo no puede pasar de 1
            with self.assertRaises(ValueError, msg=str(caso)) as contexto:
                config_desde_json(caso)
            self.assertIn("Techo de ocupación", str(contexto.exception))
            self.assertIn("nivel de ocupación", str(contexto.exception))

    def test_el_reparto_de_meses_pico_se_normaliza(self):
        from asrs.config import config_desde_json
        config = config_desde_json({"vector_reparto_mes_pico": [2, 1, 1]})
        self.assertEqual(config["vector_reparto_mes_pico"].tolist(), [0.5, 0.25, 0.25])

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
        # Instancia del capítulo 6: 0,5 m/s² en filas y columnas y velocidad constante en la profundidad
        self.assertEqual(config["vector_aceleracion_grua"].tolist(), [0.5, 0.5, 0.0])
        self.assertEqual(config_desde_json({"vector_aceleracion_grua": [0, 0, 0]})["vector_aceleracion_grua"].tolist(), [0, 0, 0])
        for caso in ({"vector_aceleracion_grua": [1, -1, 1]}, {"vector_deceleracion_grua": [1, 1]}, {"vector_aceleracion_grua": [1, "x", 1]}):
            with self.assertRaises(ValueError, msg=str(caso)):
                config_desde_json(caso)

    def test_tareas_por_defecto_son_el_diez_por_ciento(self):
        from asrs.config import config_desde_json
        self.assertEqual(config_desde_json({"vector_configuracion_estanteria": [10, 10, 2], "escalar_numero_skus": 20})["escalar_largo_secuencia"], 20)

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
            {"texto_criterio_clases": "skus", "escalar_numero_skus": 3, "vector_proporcion_skus_por_clase": [1, 1, 1, 1]},
            {"texto_criterio_clases": "iguales", "escalar_numero_skus": 3, "escalar_numero_clases": 4},
            {"texto_criterio_clases": "otro"},
            {"escalar_numero_clases": 0},
            {"vector_proporcion_skus_por_clase": [1, 0, 1]},
            {"vector_configuracion_estanteria": [2, 2, 2], "escalar_numero_skus": 30},
            {"vector_politicas_asignacion": []},
            {"vector_politicas_asignacion": ["inexistente"]},
            {"escalar_semilla": "abc"},
            {"vector_prob_acum_AR": [0.5, 0.9]},
            {"texto_perfil_inventario": "otro"},
            {"texto_regla_recuperacion": "lifo"},
            {"vector_politicas_recuperacion": []},
            {"vector_politicas_recuperacion": ["fcfs", "fcfs"]},
            {"texto_modelo_demanda": "otro"},
            {"escalar_bass_q": 0.01},                                   # q no supera a p
            {"escalar_bass_p": 0},
            {"escalar_bass_q": 6},
            {"escalar_meses_ciclo_vida": 0},
            {"escalar_techo_ocupacion": 0.8},                           # el techo no supera la ocupación
            {"escalar_techo_ocupacion": 1.2},
            {"escalar_horas_por_dia": 25},
            {"escalar_ciclos_por_hora": 0},
            {"escalar_ciclos_por_hora": 0.5, "escalar_horas_por_dia": 1},  # menos de una recuperación por día
            {"vector_cuantiles_amplitud": [0.2, 0.1, 0.3, 0.4, 0.5]},
            {"vector_cuantiles_amplitud": [0.1, 0.2, 0.3, 0.4]},
            {"vector_probabilidades_amplitud": [0, 0.5, 0.5, 0.9, 1]},
            {"vector_reparto_mes_pico": [1, -1, 1]},
            {"vector_reparto_mes_pico": [0, 0, 0]},
            {"escalar_dias_calentamiento": 10},                         # menor que la ventana
            {"escalar_dias_ventana_popularidad": 0},
            {"escalar_dias_ventana_popularidad": 400, "escalar_dias_calentamiento": 500},
            {"escalar_dias_calentamiento": 731},
            {"escalar_largo_secuencia": 600_001},
        ]
        for caso in invalidos:
            with self.assertRaises(ValueError, msg=str(caso)):
                config_desde_json(caso)

    def test_regla_de_recuperacion_antigua_sigue_cargando(self):
        from asrs.config import config_desde_json
        # Las configuraciones guardadas antes de admitir varias recuperaciones traían una sola en 'texto_regla_recuperacion'
        self.assertEqual(config_desde_json({})["vector_politicas_recuperacion"], ('fcfs',))
        self.assertEqual(config_desde_json({"texto_regla_recuperacion": "fcfs_accesible"})["vector_politicas_recuperacion"], ('fcfs_accesible',))
        self.assertEqual(config_desde_json({"texto_regla_recuperacion": "fcfs_accesible", "vector_politicas_recuperacion": ["fcfs"]})["vector_politicas_recuperacion"], ('fcfs',))


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

    def test_cuantil_t_de_student(self):
        from asrs.simulacion.estadistica import cuantil_t_975
        # Valores de tabla de la t de Student para el 95 % bilateral
        for escalar_gl, escalar_tabla in ((1, 12.706), (2, 4.303), (3, 3.182), (10, 2.228), (19, 2.093), (30, 2.042), (1000, 1.962)):
            self.assertAlmostEqual(cuantil_t_975(escalar_gl) / escalar_tabla, 1, delta=0.002, msg=str(escalar_gl))

    def test_comparacion_pareada_con_la_mejor(self):
        from asrs.simulacion.estadistica import resumir

        def fila(nombre, total):
            return [nombre] + [0] * 27 + [10, total]

        # B tarda siempre 10 s más que A: los intervalos de cada una se solapan, pero la diferencia réplica a réplica es constante
        resultado = resumir([[fila("A", total), fila("B", total + 10)] for total in (100, 300, 200)])
        a, b = resultado["politicas"]
        self.assertIsNone(a["tiempo_total"]["vs_mejor"])
        self.assertEqual(b["tiempo_total"]["vs_mejor"]["media"], 10)
        self.assertEqual(b["tiempo_total"]["vs_mejor"]["ic95"], 0)
        self.assertGreater(a["tiempo_total"]["media"] + a["tiempo_total"]["ic95"], b["tiempo_total"]["media"] - b["tiempo_total"]["ic95"])


if __name__ == "__main__":
    unittest.main()
