"""Pruebas (pytest) de la demanda por réplica del capítulo 8: día de cada recuperación, rotación, historial por días, popularidad móvil,
variabilidad con la Dirichlet y estimación de ω̂."""
import random
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pytest

from asrs.config import cargar_configuracion, config_a_json, config_desde_json
from asrs.generacion.datos_reales import derivar_lista_demanda, es_solo_demanda, escenario_datos_reales, leer_dias, leer_tareas
from asrs.generacion.fuentes import escalar_secuencia, escribir_secuencia, extraer_secuencia
from asrs.generacion.fuentes.base import reescalar_calendario, rotar_calendario
from asrs.generacion.fuentes.calzado import AdaptadorCalzado
from asrs.generacion.variabilidad import estimar_omega, mezclar_dirichlet, omega_del_nivel, recuperaciones_por_dia, tendencia
from asrs.simulacion.simulacion import Simulacion
from tests.test_fuentes import escribir_calzado


def filas_calzado(escalar_dias=60, escalar_referencias=8, semilla=0):
    # Líneas de pedido de calzado durante `escalar_dias` días, con demanda concentrada en las primeras referencias
    generador = np.random.default_rng(semilla)
    filas, inicio = [], datetime(2023, 1, 2, 8, 0)
    for escalar_d in range(escalar_dias):
        for _ in range(int(generador.integers(2, 6))):
            escalar_ref = int(min(generador.geometric(0.35), escalar_referencias))
            filas.append((f"R{escalar_ref:02d}", "40", int(generador.integers(1, 4)), (inicio + timedelta(days=escalar_d)).strftime("%d/%m/%Y %H:%M")))
    return filas


def csv_con_dias(vector_skus, vector_dias):
    return "SKU,Dia\n" + "".join(f"{s},{d}\n" for s, d in zip(vector_skus, vector_dias))


class TestCalendario:
    def test_escalar_conserva_el_dia_de_cada_copia(self):
        skus, dias = escalar_secuencia(["A", "B", "C"], 7, [0, 1, 1])
        assert len(skus) == len(dias) == 7
        assert all({"A": 0, "B": 1, "C": 1}[s] == d for s, d in zip(skus, dias))
        assert escalar_secuencia(["A", "B"], 4) == ["A", "A", "B", "B"]       # sin días la salida no cambia

    def test_reescalar_calendario_cubre_los_dias_del_modelo(self):
        dias = reescalar_calendario(list(range(10)), 10, 395)
        assert dias[0] == 0 and dias[-1] == 9 * 395 // 10 and dias == sorted(dias) and max(dias) < 395

    def test_rotacion_conserva_el_orden_dentro_del_dia(self):
        dias = [0, 0, 1, 2, 2, 2, 3]
        orden, rotados, k = rotar_calendario(dias, 4, 7)
        assert sorted(orden.tolist()) == list(range(7)) and list(rotados) == sorted(rotados)
        assert all((dias[i] - k) % 4 == r for i, r in zip(orden.tolist(), rotados.tolist()))
        # Dentro de cada día las recuperaciones siguen en su orden original
        for escalar_dia in set(rotados.tolist()):
            indices = [i for i, r in zip(orden.tolist(), rotados.tolist()) if r == escalar_dia]
            assert indices == sorted(indices)
        assert rotar_calendario(dias, 4, 7)[2] == k                         # reproducible con la semilla

    def test_extraer_con_dias_y_rotacion(self, tmp_path):
        ruta = escribir_calzado(tmp_path / "c.csv", filas_calzado())
        sin_rotar = extraer_secuencia(AdaptadorCalzado(), ruta, None, 0, True, 500, 395, False)
        assert len(sin_rotar.vector_dias) == 500 and sin_rotar.vector_dias == sorted(sin_rotar.vector_dias)
        assert sin_rotar.vector_dias[0] == 0 and max(sin_rotar.vector_dias) < 395 and sin_rotar.escalar_dias_fuente == 60
        desplazamientos = {extraer_secuencia(AdaptadorCalzado(), ruta, None, s, True, 500, 395, True).escalar_desplazamiento for s in range(6)}
        assert len(desplazamientos) > 1                                       # cada semilla empieza en otro tramo
        rotada = extraer_secuencia(AdaptadorCalzado(), ruta, None, 3, True, None, 395, True)
        assert rotada.vector_dias == sorted(rotada.vector_dias) and rotada.escalar_desplazamiento > 0
        # Antes del escalado la rotación no cambia la demanda de cada SKU
        assert Counter(rotada.vector_skus) == Counter(extraer_secuencia(AdaptadorCalzado(), ruta, None, 3, True, None, 395, False).vector_skus)


class TestListaConDias:
    def test_csv_sku_dia(self, tmp_path):
        ruta = tmp_path / "d.csv"
        escribir_secuencia(["A", "B,1", "A"], ruta, [0, 0, 3])
        texto = ruta.read_text(encoding="utf-8")
        assert texto.splitlines()[0] == "SKU,Dia"
        assert leer_tareas(texto) == [(1, "A"), (1, "B,1"), (1, "A")] and leer_dias(texto) == [0, 0, 3]
        assert es_solo_demanda(texto) and leer_dias("SKU\nA\n") is None
        with pytest.raises(ValueError):
            leer_tareas("SKU,Dia\nA,2\nB,1\n")                                # los días deben estar en orden

    def test_historial_por_volumen_a_mitad_de_dia(self):
        # Si la recuperación R·B cae a mitad de un día, el resto de ese día es el día 0 de la operación
        cfg = derivar_lista_demanda([(1, s) for s in "AABBC"], [0, 1, 1, 2, 2], True, 50, 2, 4, 6)
        assert cfg["escalar_tareas_historial"] == 2 and cfg["escalar_dia_inicio_operacion"] == 1
        assert cfg["vector_dias_operacion"][0] == 0 and cfg["vector_dias_operacion"][-1] == 1

    def test_historial_por_dias_y_popularidad(self):
        skus = ["A"] * 6 + ["B"] * 2 + ["A", "B", "B", "C"]
        dias = [0, 0, 1, 1, 2, 2, 2, 2] + [3, 3, 4, 5]
        cfg = dict(derivar_lista_demanda([(1, s) for s in skus], dias, True, 50, 8, 4, 6), escalar_dias_ventana_popularidad=3)
        assert cfg["escalar_tareas_historial"] == 8                           # las primeras R·B = 8 recuperaciones (días 0 a 2)
        assert cfg["escalar_dia_inicio_operacion"] == 3
        assert cfg["etiquetas_sku"][:2] == ["A", "B"] and cfg["vector_demanda_sku"][2] == 0
        assert len(cfg["vector_dias_operacion"]) == len(cfg["secuencia_real"])
        assert cfg["vector_dias_operacion"][0] == 0 and cfg["vector_dias_operacion"][-1] == 2
        escenario = escenario_datos_reales(cfg, [])
        # Pop(0) son las recuperaciones de los W = B días de historial: proporcional a la demanda con que planifican las políticas
        assert escenario.popularidad(0).tolist() == [6, 2, 0]
        assert escenario.popularidad(1).tolist() == [5, 3, 0]                 # días 1 a 3
        assert escenario.activos(0).all() and escenario.vector_dia_de_tarea == cfg["vector_dias_operacion"]


class TestDirichlet:
    def secuencia_base(self, escalar_dias=200, escalar_por_dia=60, semilla=0):
        generador = np.random.default_rng(semilla)
        vector_p = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
        skus, dias = [], []
        for d in range(escalar_dias):
            conteo = generador.multinomial(escalar_por_dia, vector_p)
            for i, c in enumerate(conteo):
                skus += [f"S{i}"] * int(c)
                dias += [d] * int(c)
        return skus, dias

    def test_conserva_el_volumen_diario(self):
        skus, dias = self.secuencia_base(30)
        nuevos, nuevos_dias = mezclar_dirichlet(skus, dias, 0.3, 30, np.random.default_rng(1))
        assert Counter(nuevos_dias) == Counter(dias) and nuevos_dias == sorted(nuevos_dias)
        assert nuevos != skus

    def test_extremos(self):
        skus, dias = self.secuencia_base(20)
        unos, unos_dias = mezclar_dirichlet(skus, dias, 1.0, 30, np.random.default_rng(2))
        por_dia = {}
        for s, d in zip(unos, unos_dias):
            por_dia.setdefault(d, set()).add(s)
        assert all(len(v) == 1 for v in por_dia.values())                    # con ω = 1 cada día va entero a un SKU
        ceros, ceros_dias = mezclar_dirichlet(skus, dias, 0.0, 30, np.random.default_rng(3))
        assert estimar_omega(recuperaciones_por_dia(ceros, ceros_dias)[0], 30) < 0.02

    def test_estima_el_omega_con_que_se_genero(self):
        skus, dias = self.secuencia_base()
        for escalar_omega in (0.05, 0.2, 0.5):
            nuevos, nuevos_dias = mezclar_dirichlet(skus, dias, escalar_omega, 30, np.random.default_rng(4))
            assert estimar_omega(recuperaciones_por_dia(nuevos, nuevos_dias)[0], 30) == pytest.approx(escalar_omega, abs=0.04)

    def test_niveles(self):
        assert omega_del_nivel("baja", 0.2) == 0.1 and omega_del_nivel("real", 0.2) == 0.2 and omega_del_nivel("alta", 0.2) == 0.6
        assert omega_del_nivel("ninguna", 0.2) is None
        with pytest.raises(ValueError):
            omega_del_nivel("media", 0.2)

    def test_tendencia_centrada(self):
        matriz = np.array([[1, 0], [0, 1], [0, 1], [3, 0]])
        rho = tendencia(matriz, 2)                                            # ventana de los días d - 1 y d, desplazada hacia adentro en los bordes
        assert rho[0].tolist() == rho[1].tolist() == [0.5, 0.5] and rho[2].tolist() == [0.0, 1.0] and rho[3].tolist() == [0.75, 0.25]


class TestReplicas:
    def config_pequena(self, **extra):
        datos = config_a_json(cargar_configuracion())
        datos.update(vector_configuracion_estanteria=[6, 6, 2], escalar_ciclos_por_hora=1, escalar_horas_por_dia=4, escalar_semilla=1,
                     vector_politicas_asignacion=["aleatoria", "basada_en_clases", "full_turnover"], vector_politicas_recuperacion=["fcfs"])
        datos.update(extra)
        return config_desde_json(datos)

    def test_fuente_cambia_con_la_semilla(self, tmp_path):
        carpeta = tmp_path / "datos" / "calzado-furlan-de-assis-2024" / "crudo"
        carpeta.mkdir(parents=True)
        escribir_calzado(carpeta / "Customer_Order.csv", filas_calzado())
        cfg = self.config_pequena(texto_origen_tareas="fuente", texto_fuente_demanda="calzado", texto_carpeta_datos=str(tmp_path / "datos"),
                                  texto_variabilidad="real")
        # R = 2 recuperaciones por día durante Y + B = 395 días
        assert sum(1 for a, _ in cfg["secuencia_real"] if a == 1) + cfg["escalar_tareas_historial"] == 2 * 395
        assert cfg["dic_resumen_demanda"]["omega"] == cfg["dic_resumen_demanda"]["omega_fuente"]
        resumenes = []
        for escalar_semilla in (1, 2, 1):
            simulacion = Simulacion(cfg)
            random.seed(escalar_semilla)
            SA, S = simulacion.generarSecuencias(escalar_semilla)
            assert simulacion.escenario_demanda is not None and len(simulacion.escenario_demanda.vector_dia_de_tarea) == len(S)
            resumenes.append((S, simulacion.config["dic_resumen_demanda"]["desplazamiento_dias"], simulacion.config["dic_resumen_demanda"]["omega_fuente"]))
        assert resumenes[0] == resumenes[2]                                   # la misma semilla da la misma lista
        assert resumenes[0][0] != resumenes[1][0]                             # otra semilla, otra lista
        assert resumenes[0][2] == resumenes[1][2]                             # pero el mismo ω̂: es uno por fuente
        filas, resumen = Simulacion.cargarReplicaConEscenario((cfg, 2))
        assert len(filas) == 3 and resumen["fuente"] == "calzado" and resumen["variabilidad"] == "real"

    def test_fuente_sin_archivo(self, tmp_path):
        with pytest.raises(ValueError, match="archivo crudo"):
            self.config_pequena(texto_origen_tareas="fuente", texto_fuente_demanda="calzado", texto_carpeta_datos=str(tmp_path))

    def test_csv_con_dias_y_variabilidad(self):
        skus, dias = TestDirichlet().secuencia_base(395, 2)
        texto = csv_con_dias(skus, dias)
        fija = self.config_pequena(texto_origen_tareas="datos", texto_tareas_csv=texto)
        # Sin variabilidad la lista es la misma en todas las réplicas
        simulacion = Simulacion(fija)
        assert simulacion.generarSecuencias(5)[1] == Simulacion(fija).generarSecuencias(1)[1]
        assert "omega_fuente" in fija["dic_resumen_demanda"] and fija["escalar_tareas_historial"] == 2 * 30   # R·B
        # La popularidad del día 0 es proporcional a la demanda del historial
        pop = simulacion.escenario_demanda.popularidad(0)
        assert np.allclose(pop / pop.sum(), fija["vector_demanda_sku"])
        alta = self.config_pequena(texto_origen_tareas="datos", texto_tareas_csv=texto, texto_variabilidad="alta")
        assert Simulacion(alta).generarSecuencias(5)[1] != Simulacion(alta).generarSecuencias(1)[1]
        with pytest.raises(ValueError, match="día de cada recuperación"):
            self.config_pequena(texto_origen_tareas="datos", texto_tareas_csv="SKU\n" + "\n".join(skus), texto_variabilidad="real")
