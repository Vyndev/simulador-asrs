"""Pruebas (pytest) del tratamiento de fuentes: orden, empates, filtros, salida y lectura de la salida con `leer_tareas`."""
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
import pytest

from asrs.generacion.datos_reales import leer_tareas, preparar_datos_reales
from asrs.generacion.fuentes import ADAPTADORES, Adaptador, Registro, codificar_skus, crear_adaptador, escribir_codigos, escribir_secuencia, escalar_secuencia, extraer_secuencia, muestrear_surtido
from asrs.generacion.fuentes.calzado import ENCABEZADO as ENCABEZADO_CALZADO, AdaptadorCalzado
from asrs.generacion.fuentes.ecommerce import ENCABEZADO as ENCABEZADO_ECOMMERCE, AdaptadorEcommerce


def escribir_calzado(ruta, filas):
    # filas: (referencia, talla, cantidad, "dd/mm/aaaa hh:mm"); el archivo real tiene BOM, CRLF y espacios en el cliente
    lineas = [";".join(ENCABEZADO_CALZADO)]
    for referencia, talla, cantidad, fecha in filas:
        lineas.append(f"C001            ;1;1;{referencia};{talla};{cantidad};{fecha};7;Operator_1")
    ruta.write_bytes(("﻿" + "\r\n".join(lineas) + "\r\n").encode("utf-8"))
    return str(ruta)


def serie_excel(fecha_iso):
    # días desde 1899-12-30, como los guarda Excel
    return (datetime.fromisoformat(fecha_iso) - datetime(1899, 12, 30)).total_seconds() / 86400


def escribir_xlsx(ruta, filas):
    # filas: (fecha ISO o "NA", sku, cantidad). Genera un .xlsx mínimo con cadenas compartidas, como el original
    textos = []

    def cadena(txt):
        if txt not in textos:
            textos.append(txt)
        return textos.index(txt)

    xml_filas = []
    encabezado = "".join(f'<c r="{"ABCDEF"[i]}1" t="s"><v>{cadena(n)}</v></c>' for i, n in enumerate(ENCABEZADO_ECOMMERCE))
    xml_filas.append(f'<row r="1">{encabezado}</row>')
    for k, (fecha, sku, cantidad) in enumerate(filas, start=2):
        celdas = [f'<c r="A{k}"><v>{k}</v></c>', f'<c r="B{k}" t="s"><v>{cadena("O" + str(k))}</v></c>']
        celdas.append(f'<c r="C{k}" t="s"><v>{cadena("NA")}</v></c>' if fecha == "NA" else f'<c r="C{k}"><v>{serie_excel(fecha)!r}</v></c>')
        celdas.append(f'<c r="D{k}" t="s"><v>{cadena("NA")}</v></c>')
        celdas.append(f'<c r="E{k}" t="s"><v>{cadena(sku)}</v></c>')
        celdas.append(f'<c r="F{k}"><v>{cantidad}</v></c>')
        xml_filas.append(f'<row r="{k}">{"".join(celdas)}</row>')
    espacio = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{espacio}">' + "".join(f"<si><t>{escape(t)}</t></si>" for t in textos) + "</sst>")
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{espacio}"><sheetData>{"".join(xml_filas)}</sheetData></worksheet>')
    return str(ruta)


class TestNucleo:
    def test_ordena_por_tiempo_y_los_empates_conservan_el_orden_del_archivo(self, tmp_path):
        # El archivo viene en desorden de tiempo y con empates en el mismo minuto
        ruta = escribir_calzado(tmp_path / "c.csv", [
            ("B", "9", 1, "02/01/2023 07:00"),
            ("Z", "9", 1, "01/01/2023 07:00"),
            ("A", "9", 1, "02/01/2023 07:00"),
            ("Y", "9", 1, "01/01/2023 07:00"),
            ("C", "9", 1, "01/01/2023 06:59"),
        ])
        resultado = extraer_secuencia(AdaptadorCalzado(), ruta, None)
        assert resultado.vector_skus == ["C", "Z", "Y", "B", "A"]

    def test_adaptador_nuevo_solo_define_filas_e_interpretar(self):
        class Simple(Adaptador):
            def filas(self, ruta):
                return iter([("x", 3), ("y", 1), ("n", 2), ("z", 1)])

            def interpretar(self, fila):
                return Registro(fila[0], fila[1]) if fila[0] != "n" else Registro(txt_motivo="ruido")

        resultado = extraer_secuencia(Simple(), "ignorada")
        assert resultado.vector_skus == ["y", "z", "x"]
        assert resultado.dic_descartes == {"ruido": 1}
        assert (resultado.escalar_filas_leidas, resultado.escalar_descartadas, resultado.escalar_skus_distintos) == (4, 1, 3)

    def test_registro(self):
        assert set(ADAPTADORES) == {"calzado", "ecommerce", "inventario_sintetico"}
        assert isinstance(crear_adaptador("calzado"), AdaptadorCalzado)
        with pytest.raises(ValueError):
            crear_adaptador("inexistente")


def secuencia_sintetica():
    # 3 SKU pesados (60, 50, 40 tareas) y una cola de 30 SKU de 1 a 3 tareas; 6 etiquetas como máximo
    pesados = ["P1"] * 60 + ["P2"] * 50 + ["P3"] * 40
    cola = [f"c{i:02d}" for i in range(30) for _ in range(1 + i % 3)]
    return [x for par in zip(pesados, pesados[::-1]) for x in par][:0] + pesados + cola


class TestMuestreo:
    def test_individuales_cabeza_y_muestra_de_la_cola_hasta_completar_el_maximo(self):
        secuencia = secuencia_sintetica()
        reducida, dic, conteos = muestrear_surtido(secuencia, 6, 0, 0.3)       # umbral: 0,3 x 210 / 6 = 10,5 tareas
        assert [k for k, v in dic.items() if v == "individual"] == ["P1", "P2", "P3"]
        assert conteos == {"individuales": 3, "cola": 30, "muestreados": 3}
        muestra = {k for k, v in dic.items() if v == "muestra"}
        assert len(muestra) == 3 and set(reducida) == {"P1", "P2", "P3"} | muestra
        assert all(dic[k] == "descartado" for k in dic if k not in muestra and not k.startswith("P"))

    def test_conserva_el_orden_relativo_y_las_rachas_de_los_elegidos(self):
        secuencia = ["A"] * 5 + ["x"] + ["B"] * 4 + ["y", "y"] + ["A"] * 3
        reducida, dic, _ = muestrear_surtido(secuencia, 2, 0, 0.1)
        assert dic["A"] == "individual" and dic["B"] == "individual" or dic["A"] == "individual"       # A manda; el otro cupo es de la cola
        elegidos = [k for k, v in dic.items() if v != "descartado"]
        assert reducida == [x for x in secuencia if x in elegidos]
        assert "AAAAA" in "".join(reducida) and "AAA" in "".join(reducida)

    def test_es_reproducible_con_la_misma_semilla_y_cambia_con_otra(self):
        secuencia = secuencia_sintetica()
        assert muestrear_surtido(secuencia, 6, 7) == muestrear_surtido(secuencia, 6, 7)
        muestras = {tuple(sorted(k for k, v in muestrear_surtido(secuencia, 6, sem)[1].items() if v == "muestra")) for sem in range(8)}
        assert len(muestras) > 1

    def test_no_muestrea_si_el_surtido_ya_cabe(self):
        secuencia = ["a", "b", "a", "c"]
        assert muestrear_surtido(secuencia, 5) == (secuencia, {"a": "individual", "b": "individual", "c": "individual"},
                                                   {"individuales": 3, "cola": 0, "muestreados": 0})

    def test_a_lo_sumo_maximo_menos_uno_individuales_y_siempre_queda_cupo_para_la_cola(self):
        secuencia = [f"s{i}" for i in range(10) for _ in range(10)]               # todos superan cualquier umbral bajo
        reducida, dic, conteos = muestrear_surtido(secuencia, 4, 0, 0.01)
        assert conteos["individuales"] == 3 and conteos["muestreados"] == 1 and len(set(reducida)) == 4

    def test_extraer_reduce_en_calzado_y_ecommerce_pero_no_con_none(self, tmp_path):
        assert AdaptadorEcommerce.escalar_max_skus == 180 and AdaptadorCalzado.escalar_max_skus == 180
        filas = [("2015-11-03 10:%02d:00" % i, f"S{i}", 1) for i in range(20) for _ in range(i % 4 + 1)]
        ruta = escribir_xlsx(tmp_path / "e.xlsx", filas)
        r = extraer_secuencia(AdaptadorEcommerce(), ruta, 8)
        assert len(set(r.vector_skus)) == 8 and set(r.dic_estratos) == {f"S{i}" for i in range(20)}
        assert r.escalar_recuperaciones_antes == 50 and len(r.vector_skus) < 50
        assert len(extraer_secuencia(AdaptadorEcommerce(), ruta, None).vector_skus) == 50


class TestDesagregacion:
    def test_una_linea_de_cantidad_q_son_q_recuperaciones_seguidas_en_su_posicion(self, tmp_path):
        ruta = escribir_calzado(tmp_path / "c.csv", [
            ("B", "9", 3, "02/01/2023 07:00"),
            ("A", "9", 2, "01/01/2023 07:00"),
            ("C", "9", 1, "02/01/2023 07:00"),      # empate con B: B va antes por el archivo
            ("A", "9", 1, "01/01/2023 07:00"),      # empate con la primera A
        ])
        resultado = extraer_secuencia(AdaptadorCalzado(), ruta, None)
        assert resultado.vector_skus == ["A", "A", "A", "B", "B", "B", "C"]
        assert (resultado.escalar_lineas, len(resultado.vector_skus)) == (4, 7)

    def test_las_lineas_grandes_forman_rachas_y_se_reportan(self, tmp_path):
        ruta = escribir_calzado(tmp_path / "c.csv", [("X", "9", 40, "01/01/2023 07:00"), ("Y", "9", 1, "01/01/2023 07:01"),
                                                      ("X", "9", 2, "01/01/2023 07:02")])
        resultado = extraer_secuencia(AdaptadorCalzado(), ruta, None)
        assert resultado.vector_skus == ["X"] * 40 + ["Y"] + ["X"] * 2
        assert resultado.vector_lineas_mayores[0] == (40, "X", datetime(2023, 1, 1, 7, 0))
        assert [q for q, _, _ in resultado.vector_lineas_mayores] == [40, 2, 1]
        assert resultado.escalar_repeticiones_consecutivas == pytest.approx(40 / 42)      # 39 repeticiones en la primera racha y 1 en la segunda, de 42 transiciones

    def test_cantidad_cero_negativa_o_ilegible_se_descarta_con_su_motivo(self, tmp_path):
        ruta = escribir_calzado(tmp_path / "c.csv", [("A", "9", 0, "01/01/2023 07:00"), ("B", "9", -2, "01/01/2023 07:00"),
                                                      ("C", "9", "x", "01/01/2023 07:00"), ("D", "9", "3.0", "01/01/2023 07:00")])
        resultado = extraer_secuencia(AdaptadorCalzado(), ruta, None)
        assert resultado.vector_skus == ["D"] * 3
        assert resultado.dic_descartes == {"cantidad igual a cero": 1, "cantidad negativa": 1, "cantidad ilegible": 1}

    def test_una_cantidad_no_entera_es_un_error_y_no_se_redondea(self, tmp_path):
        ruta = escribir_calzado(tmp_path / "c.csv", [("A", "9", "1.5", "01/01/2023 07:00")])
        with pytest.raises(ValueError, match="no entera"):
            extraer_secuencia(AdaptadorCalzado(), ruta, None)
        from asrs.generacion.fuentes import leer_cantidad
        assert leer_cantidad("12") == (12, None) and leer_cantidad("0") == (None, "cantidad igual a cero")

    def test_el_muestreo_se_calcula_sobre_las_unidades(self, tmp_path):
        # B tiene una sola línea de 50 unidades y A cinco líneas de 1: por unidades B pesa más y con el tope por defecto queda individual
        filas = [("2015-11-03 10:0%d:00" % i, "A", 1) for i in range(5)] + [("2015-11-03 10:09:00", "B", 50)] + \
                [("2015-11-03 11:%02d:00" % i, f"C{i}", 1) for i in range(10)]
        ruta = escribir_xlsx(tmp_path / "e.xlsx", filas)
        r = extraer_secuencia(AdaptadorEcommerce(), ruta, 4)
        assert r.dic_estratos["B"] == "individual" and r.vector_skus.count("B") == 50


class TestCalzado:
    FILAS = [
        ("R1", "9", 2, "03/01/2023 07:00"),
        ("R1", "10", 1, "03/01/2023 07:00"),
        ("R2", "", 1, "03/01/2023 07:00"),
        ("R3", "8", 0, "03/01/2023 07:00"),     # cantidad cero: se descarta
        ("R1", "9", 5, "04/01/2023 07:00"),
    ]

    def test_el_sku_por_defecto_es_la_referencia_y_filtra_cantidad_cero(self, tmp_path):
        resultado = extraer_secuencia(AdaptadorCalzado(), escribir_calzado(tmp_path / "c.csv", self.FILAS), None)
        assert resultado.vector_skus == ["R1"] * 3 + ["R2"] + ["R1"] * 5      # una recuperación por unidad; la talla vacía no excluye con SKU = referencia
        assert resultado.dic_descartes == {"cantidad igual a cero": 1}
        assert resultado.escalar_filas_leidas == 5

    def test_sku_con_talla_es_configurable_y_excluye_talla_vacia(self, tmp_path):
        resultado = extraer_secuencia(AdaptadorCalzado(bool_sku_con_talla=True), escribir_calzado(tmp_path / "c.csv", self.FILAS), None)
        assert resultado.vector_skus == ["R1-9", "R1-9", "R1-10"] + ["R1-9"] * 5
        assert resultado.dic_descartes == {"cantidad igual a cero": 1, "talla vacía (SKU con talla)": 1}

    def test_encabezado_inesperado_falla(self, tmp_path):
        ruta = tmp_path / "c.csv"
        ruta.write_text("a;b\n1;2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            extraer_secuencia(AdaptadorCalzado(), str(ruta))


class TestEcommerce:
    def test_filtra_fechas_fuera_de_campana_cantidad_cero_y_sku_vacio_y_recorta_espacios(self, tmp_path):
        ruta = escribir_xlsx(tmp_path / "e.xlsx", [
            ("2015-11-03 10:00:00", "B", 1),
            ("2015-11-03 09:00:00", " A ", 2),         # espacios: se recortan
            ("2011-05-01 10:00:00", "VIEJO", 1),        # fuera de la campaña
            ("2026-01-01 10:00:00", "FUTURO", 1),
            ("2015-07-31 23:59:59", "ANTES", 1),
            ("2015-12-31 23:59:59", "FIN", 1),          # último instante dentro de la ventana
            ("2015-11-04 10:00:00", "CERO", 0),
            ("2015-11-04 10:00:00", "  ", 1),           # SKU vacío tras recortar
            ("NA", "SINFECHA", 1),
        ])
        resultado = extraer_secuencia(AdaptadorEcommerce(), ruta)
        assert resultado.vector_skus == ["A", "A", "B", "FIN"]      # A tiene cantidad 2
        assert resultado.dic_descartes == {"fecha de pedido fuera de la campaña": 3, "cantidad igual a cero": 1, "SKU vacío": 1,
                                           "fecha de pedido ausente": 1}

    def test_empates_por_orden_del_archivo_aunque_la_serie_de_excel_tenga_ruido_de_coma_flotante(self, tmp_path):
        ruta = escribir_xlsx(tmp_path / "e.xlsx", [("2015-11-03 10:00:00", s, 1) for s in ("Q", "P", "R")])
        assert extraer_secuencia(AdaptadorEcommerce(), ruta).vector_skus == ["Q", "P", "R"]

    def test_ventana_configurable(self, tmp_path):
        ruta = escribir_xlsx(tmp_path / "e.xlsx", [("2015-09-01 00:00:00", "S", 1), ("2015-11-01 00:00:00", "N", 1)])
        resultado = extraer_secuencia(AdaptadorEcommerce("2015-10-01", "2015-12-01"), ruta)
        assert resultado.vector_skus == ["N"]


def escribir_zip(ruta, nombre_interno, encabezado, filas):
    contenido = "\ufeff" + ",".join(encabezado) + "\n" + "\n".join(",".join(f) for f in filas) + "\n"
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr(nombre_interno, contenido)
    return str(ruta)


class TestFuentesZip:
    def test_inventario_sintetico_ordena_por_fecha_y_descarta_sin_ventas(self, tmp_path):
        base = ["WH_1", "SUP_1", "West", "{u}", "500", "10", "300", "100", "10", "15", "0", "0", "9"]
        def fila(fecha, sku, u, alm="WH_1"):
            return [fecha, sku, alm] + [x.format(u=u) for x in base[1:]]
        from asrs.generacion.fuentes.inventario_sintetico import ENCABEZADO, AdaptadorInventarioSintetico
        ruta = escribir_zip(tmp_path / "i.zip", "supply_chain_dataset1.csv", ENCABEZADO, [
            fila("2024-01-02", "SKU_1", 5), fila("2024-01-01", "SKU_2", 7), fila("2024-01-01", "SKU_1", 0),
            fila("2024-01-01", "SKU_3", 3, "WH_2")])
        r = extraer_secuencia(AdaptadorInventarioSintetico(), ruta)
        assert r.vector_skus == ["SKU_2"] * 7 + ["SKU_3"] * 3 + ["SKU_1"] * 5 and r.dic_descartes == {"sin ventas (unidades cero)": 1}
        r = extraer_secuencia(AdaptadorInventarioSintetico("WH_1"), ruta)
        assert r.vector_skus == ["SKU_2"] * 7 + ["SKU_1"] * 5 and r.dic_descartes["otro almacén"] == 1


class TestCodificacion:
    def test_codigos_por_demanda_decreciente_y_orden_intacto(self):
        secuencia = ["zeta", "x-9", "zeta", "q", "x-9", "zeta"]
        codificada, dic = codificar_skus(secuencia)
        assert dic == {"zeta": "SKU001", "x-9": "SKU002", "q": "SKU003"}
        assert codificada == ["SKU001", "SKU002", "SKU001", "SKU003", "SKU002", "SKU001"]

    def test_la_correspondencia_cubre_todos_los_sku_originales_con_su_estrato(self, tmp_path):
        filas = [("2015-11-03 10:%02d:00" % i, f"S{i}", 1) for i in range(20) for _ in range(i % 4 + 1)]
        ruta = escribir_xlsx(tmp_path / "e.xlsx", filas)
        r = extraer_secuencia(AdaptadorEcommerce(), ruta, 8, 0, True)
        destino = tmp_path / "skus.csv"
        escribir_codigos(r.dic_estratos, r.dic_codigos, str(destino))
        import csv as _csv
        lineas = list(_csv.reader(destino.open(encoding="utf-8")))
        assert lineas[0] == ["SKU_original", "SKU", "estrato"] and len(lineas) == 21
        assert {x[0] for x in lineas[1:]} == {f"S{i}" for i in range(20)}
        assert {x[2] for x in lineas[1:]} <= {"individual", "muestra", "descartado"}
        assert all((x[1] == "") == (x[2] == "descartado") for x in lineas[1:])
        assert {x[1] for x in lineas[1:] if x[1]} == set(r.vector_skus) and len(set(r.vector_skus)) == 8


class TestEscalado:
    def test_total_exacto_para_muchos_n_y_N(self):
        for n in (1, 2, 7, 10, 101, 997):
            secuencia = [f"s{i % 5}" for i in range(n)]
            for N in (1, 3, 10, 100, 1000, 1001):
                assert len(escalar_secuencia(secuencia, N)) == N

    def test_a_la_baja_cada_recuperacion_aparece_0_o_1_veces_y_conserva_el_orden(self):
        secuencia = [f"s{i:03d}" for i in range(100)]
        escalada = escalar_secuencia(secuencia, 40)
        assert len(escalada) == 40 and escalada == sorted(set(escalada)) and set(escalada) <= set(secuencia)
        posiciones = [secuencia.index(x) for x in escalada]
        assert posiciones == sorted(posiciones)
        assert max(posiciones[i + 1] - posiciones[i] for i in range(39)) <= 3          # adelgazamiento uniforme: sin huecos grandes

    def test_al_alza_cada_recuperacion_aparece_floor_o_ceil_veces_seguidas(self):
        secuencia = ["a", "a", "b", "c", "c", "c"]
        escalada = escalar_secuencia(secuencia, 15)                                      # factor 2,5: cada una sale 2 o 3 veces
        assert len(escalada) == 15
        import itertools
        rachas = [(k, len(list(g))) for k, g in itertools.groupby(escalada)]
        assert [k for k, _ in rachas] == ["a", "b", "c"]                                 # el orden se conserva
        assert {k: sum(q for kk, q in rachas if kk == k) for k, _ in rachas} == {"a": 5, "b": 3, "c": 7}
        assert escalar_secuencia(["a", "b"], 6) == ["a"] * 3 + ["b"] * 3
        assert escalar_secuencia(["a", "b", "c"], 3) == ["a", "b", "c"]

    def test_conserva_las_proporciones_de_cada_sku_hasta_una_unidad_de_redondeo(self):
        secuencia = ["a"] * 300 + ["b"] * 100 + ["c"] * 50 + ["a"] * 50
        for N in (97, 450, 1000, 4321):
            escalada = escalar_secuencia(secuencia, N)
            for sku in "abc":
                esperado = secuencia.count(sku) * N / len(secuencia)
                assert abs(escalada.count(sku) - esperado) <= 1.5

    def test_extraer_escala_despues_de_muestrear_y_antes_de_codificar(self, tmp_path):
        filas = [("2015-11-03 10:%02d:00" % i, f"S{i}", 1) for i in range(20) for _ in range(i % 4 + 1)]
        ruta = escribir_xlsx(tmp_path / "e.xlsx", filas)
        r = extraer_secuencia(AdaptadorEcommerce(), ruta, 8, 0, True, 0.3, 100)
        assert len(r.vector_skus) == 100 and r.escalar_total == 100 and r.escalar_recuperaciones_muestreadas < 50
        assert set(r.vector_skus) <= {f"SKU{i:03d}" for i in range(1, 9)}
        cuenta = {k: r.vector_skus.count(k) for k in set(r.vector_skus)}
        assert cuenta["SKU001"] == max(cuenta.values())                                  # los códigos siguen la demanda de la secuencia final
        assert len(extraer_secuencia(AdaptadorEcommerce(), ruta, 8, 0, True, 0.3, None).vector_skus) == r.escalar_recuperaciones_muestreadas

    def test_un_sku_que_pierde_todas_sus_copias_se_marca_sin_copias(self, tmp_path):
        filas = [("2015-11-03 10:00:00", "G", 100), ("2015-11-03 10:01:00", "mini", 1), ("2015-11-03 10:02:00", "H", 100)]
        ruta = escribir_xlsx(tmp_path / "e.xlsx", filas)
        r = extraer_secuencia(AdaptadorEcommerce(), ruta, None, 0, True, 0.3, 20)       # factor 0,1: la línea de 1 unidad se queda sin copias
        assert r.set_sin_copias == {"mini"} and len(r.vector_skus) == 20
        destino = tmp_path / "skus.csv"
        escribir_codigos(r.dic_estratos, r.dic_codigos, str(destino), r.set_sin_copias)
        import csv as _csv
        lineas = list(_csv.reader(destino.open(encoding="utf-8")))
        assert lineas[0] == ["SKU_original", "SKU", "estrato", "sin_copias"]
        assert [x for x in lineas[1:] if x[0] == "mini"] == [["mini", "", "individual", "sin_copias"]]
        assert all(x[3] == "" and x[1] != "" for x in lineas[1:] if x[0] != "mini")


class TestModoDatosReales:
    def csv_demanda(self, n=1200, skus=12):
        generador = np.random.default_rng(5)
        return [f"P{generador.integers(1, skus + 1)}" for _ in range(n)]

    def config_datos(self, texto_csv):
        from asrs.config import cargar_configuracion, config_a_json, config_desde_json
        datos = config_a_json(cargar_configuracion())
        datos.update(texto_origen_tareas="datos", texto_tareas_csv=texto_csv, escalar_porcentaje_historial=100, escalar_cantidad_replicas=1,
                     vector_politicas_asignacion=["mas_cercana_grua"])
        return config_desde_json(datos)

    def test_constantes_de_la_instancia(self):
        from asrs.config import RECUPERACIONES_POR_CORRIDA, TAREAS_POR_CORRIDA, cargar_configuracion, recuperaciones_por_corrida
        assert recuperaciones_por_corrida(cargar_configuracion()) == RECUPERACIONES_POR_CORRIDA == TAREAS_POR_CORRIDA // 2 == 262_800
        assert 2 * RECUPERACIONES_POR_CORRIDA == TAREAS_POR_CORRIDA

    def test_la_reposicion_eoq_entra_en_lote_justo_despues_de_agotar_el_sku(self):
        import math
        from asrs.generacion.datos_reales import reponer_con_lotes_eoq
        from asrs.generacion.reposicion import calibrar_kappa, lotes_eoq
        recuperaciones = [(1, 1 + (i * 7) % 3) for i in range(600)]               # 3 SKU con demanda 1/3 cada uno
        demanda = np.array([1 / 3] * 3)
        operacion, inicial = reponer_con_lotes_eoq(recuperaciones, demanda, 720, 30)
        nu = 720 * demanda
        lote = lotes_eoq(nu, calibrar_kappa(nu, 30)).tolist()
        assert inicial == [math.ceil(q / 2) for q in lote] and sum(inicial) <= 30 + 3
        assert [t for t in operacion if t[0] == 1] == recuperaciones               # las recuperaciones, en el mismo orden
        existencias = list(inicial)
        i = 0
        while i < len(operacion):
            actividad, sku = operacion[i]
            assert actividad == 1
            existencias[sku - 1] -= 1
            assert existencias[sku - 1] >= 0
            i += 1
            if existencias[sku - 1] == 0:                                           # agotó el SKU: entra justo después el lote completo
                assert operacion[i:i + lote[sku - 1]] == [(0, sku)] * lote[sku - 1]
                existencias[sku - 1] = lote[sku - 1]
                i += lote[sku - 1]
        assert abs(sum(1 for t in operacion if t[0] == 0) - 600) <= sum(lote)       # almacenamientos ~ recuperaciones: la estantería se mantiene

    def test_un_csv_de_solo_demanda_genera_los_almacenamientos_con_lotes_en_config(self):
        vector_skus = self.csv_demanda(2400)
        config = self.config_datos("SKU\n" + "\n".join(vector_skus) + "\n")
        operacion = config["secuencia_real"]
        assert sum(1 for actividad, _ in operacion if actividad == 1) == len(vector_skus)
        assert abs(sum(1 for actividad, _ in operacion if actividad == 0) - len(vector_skus)) < 100
        assert config["vector_inventario_inicial"] is not None and sum(config["vector_inventario_inicial"]) > 0

    def test_un_csv_con_columna_task_se_usa_tal_cual_sin_inventario_inicial_propio(self):
        config = self.config_datos("SKU;Task\nA;R\nB;R\nC;R\nA;A\n")
        assert config["secuencia_real"] == [(1, 1), (1, 2), (1, 3), (0, 1)] and config["vector_inventario_inicial"] is None

    def test_la_estanteria_no_se_vacia_con_la_reposicion_eoq(self, monkeypatch):
        from asrs.politicas import recuperacion
        from asrs.simulacion.simulacion import Simulacion
        vector_skus = self.csv_demanda()
        omitidas = {"vacias": 0, "llamadas": 0}
        original = recuperacion.PoliticaRecuperacionFCFS.seleccionar

        def contando(self, estanteria, sku):
            elegido = original(self, estanteria, sku)
            omitidas["llamadas"] += 1
            omitidas["vacias"] += 0 if len(elegido) else 1
            return elegido
        monkeypatch.setattr(recuperacion.PoliticaRecuperacionFCFS, "seleccionar", contando)

        config = self.config_datos("SKU\n" + "\n".join(vector_skus) + "\n")
        Simulacion.cargarReplica((config, 1))
        assert omitidas["llamadas"] == len(vector_skus)
        vacias_con_reposicion = omitidas["vacias"]

        # El mismo CSV con columna Task (solo recuperaciones, sin almacenamientos): la estantería se vacía y se omiten muchas
        omitidas.update(vacias=0, llamadas=0)
        config_sin = self.config_datos("SKU;Task\n" + "\n".join(f"{sku};R" for sku in vector_skus) + "\n")
        Simulacion.cargarReplica((config_sin, 1))
        assert omitidas["vacias"] > 500 > vacias_con_reposicion


class TestSalida:
    def test_escribe_una_columna_con_encabezado(self, tmp_path):
        ruta = tmp_path / "s.csv"
        escribir_secuencia(["A", "B", "A"], str(ruta))
        assert ruta.read_text(encoding="utf-8") == "SKU\nA\nB\nA\n"

    def test_leer_tareas_acepta_la_salida_como_recuperaciones(self, tmp_path):
        ruta = tmp_path / "s.csv"
        escribir_secuencia(["A", "B", "A", "SKU, con coma"], str(ruta))
        tareas = leer_tareas(ruta.read_text(encoding="utf-8"))
        assert tareas == [(1, "A"), (1, "B"), (1, "A"), (1, "SKU, con coma")]
        secuencia, demanda, etiquetas = preparar_datos_reales(tareas)
        assert etiquetas[0] == "A" and float(demanda.sum()) == pytest.approx(1.0)

    def test_leer_tareas_una_columna_limites_y_errores(self):
        assert leer_tareas("sku\r\nX\r\nY\r\n") == [(1, "X"), (1, "Y")]
        for texto in ("SKU\n", "SKU\n\"\"\n", "solo una columna"):
            with pytest.raises(ValueError):
                leer_tareas(texto)
        with pytest.raises(ValueError):
            leer_tareas("SKU\nX\nY\nZ\n", 2)
