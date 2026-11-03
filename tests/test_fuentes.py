"""Pruebas (pytest) del tratamiento de fuentes: orden, empates, filtros, salida y lectura de la salida con `leer_tareas`."""
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape

import pytest

from asrs.generacion.datos_reales import leer_tareas, preparar_datos_reales
from asrs.generacion.fuentes import ADAPTADORES, Adaptador, Registro, agrupar_surtido, codificar_skus, crear_adaptador, escribir_secuencia, extraer_secuencia
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


class TestAgrupacion:
    def test_no_quita_recuperaciones_y_reduce_el_surtido(self):
        # 6 SKU en 2 familias claras: tres muy frecuentes al inicio y tres raros al final
        secuencia = ["a", "b", "c"] * 20 + ["x", "y", "z"]
        grupos, dic = agrupar_surtido(secuencia, 2, escalar_tope=1.5)      # tope alto: ningún SKU queda solo y se ve la estructura de familias
        assert len(grupos) == len(secuencia)
        assert set(grupos) == {"G001", "G002"} and len(set(dic.values())) == 2
        assert dic["a"] == dic["b"] == dic["c"] == "G001" and dic["x"] == dic["y"] == dic["z"] == "G002"
        assert grupos[:3] == ["G001"] * 3 and grupos[-3:] == ["G002"] * 3

    def test_los_sku_pesados_quedan_individuales_y_ningun_grupo_pasa_del_tope(self):
        pesados = ["P1"] * 300 + ["P2"] * 200
        cola = [f"c{i}" for i in range(60) for _ in range(2)] + [f"u{i}" for i in range(100)]
        secuencia = [x for par in zip(pesados, pesados[::-1]) for x in par] + cola
        grupos, dic = agrupar_surtido(secuencia, 12, escalar_tope=1.5)
        assert len(grupos) == len(secuencia) and len(set(grupos)) <= 12
        assert dic["P1"] == "P1" and dic["P2"] == "P2"                # individuales: conservan su nombre
        tope = 1.5 * len(secuencia) / 12
        demanda = {g: grupos.count(g) for g in set(grupos)}
        assert all(d <= tope for g, d in demanda.items() if g.startswith("G"))

    def test_un_tope_bajo_deja_individual_la_cabeza_y_uno_alto_la_agrupa(self):
        # 12 SKU con demanda decreciente (12, 11, ..., 1 tareas): el tope fija cuántos quedan solos
        secuencia = [f"s{i}" for i in range(12) for _ in range(12 - i)]
        n, grupos_max = len(secuencia), 6
        cuantos = []
        for tope in (0.1, 0.8, 1.5):
            etiquetas, dic = agrupar_surtido(secuencia, grupos_max, 0, tope)
            individuales = {k for k, v in dic.items() if not v.startswith("G")}
            assert len(etiquetas) == n and len(set(etiquetas)) <= grupos_max
            assert all(secuencia.count(k) > tope * n / grupos_max for k in individuales)      # solo quedan solos los que pasan del tope
            cuantos.append(len(individuales))
        assert cuantos[0] == grupos_max - 1 and cuantos == sorted(cuantos, reverse=True)      # tope bajo: tantos como cupo deja a la cola
        assert cuantos[-1] < cuantos[0]

    def test_el_tope_por_defecto_es_0_1_y_extraer_lo_pasa_a_la_agrupacion(self, tmp_path):
        filas = [("2015-11-03 10:%02d:00" % i, f"S{i}", 1) for i in range(30) for _ in range(i % 5 + 1)]
        ruta = escribir_xlsx(tmp_path / "e.xlsx", filas)
        por_defecto = extraer_secuencia(AdaptadorEcommerce(), ruta, 10)
        assert por_defecto.vector_skus == extraer_secuencia(AdaptadorEcommerce(), ruta, 10, 0, False, 0.1).vector_skus
        assert por_defecto.dic_grupos != extraer_secuencia(AdaptadorEcommerce(), ruta, 10, 0, False, 1.5).dic_grupos

    def test_es_reproducible_y_no_agrupa_si_ya_cabe(self):
        secuencia = [f"s{i % 17}" for i in range(300)] + [f"r{i}" for i in range(40)]
        assert agrupar_surtido(secuencia, 5, 3) == agrupar_surtido(secuencia, 5, 3)
        assert agrupar_surtido(["a", "b", "a"], 5) == (["a", "b", "a"], {"a": "a", "b": "b"})

    def test_extraer_agrupa_por_defecto_en_ecommerce_y_no_en_calzado(self, tmp_path):
        assert AdaptadorEcommerce.escalar_grupos == 180 and AdaptadorCalzado.escalar_grupos == 180
        ruta = escribir_xlsx(tmp_path / "e.xlsx", [("2015-11-03 10:0%d:00" % i, s, 1) for i, s in enumerate("AABACB")])
        resultado = extraer_secuencia(AdaptadorEcommerce(), ruta, 2)
        assert len(resultado.vector_skus) == 6 and len(set(resultado.vector_skus)) == 2
        assert set(resultado.dic_grupos) == {"A", "B", "C"}
        assert extraer_secuencia(AdaptadorEcommerce(), ruta, None).vector_skus == list("AABACB")


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
        assert resultado.vector_skus == ["R1", "R1", "R2", "R1"]      # la talla vacía no excluye con SKU = referencia
        assert resultado.dic_descartes == {"cantidad igual a cero": 1}
        assert resultado.escalar_filas_leidas == 5

    def test_sku_con_talla_es_configurable_y_excluye_talla_vacia(self, tmp_path):
        resultado = extraer_secuencia(AdaptadorCalzado(bool_sku_con_talla=True), escribir_calzado(tmp_path / "c.csv", self.FILAS), None)
        assert resultado.vector_skus == ["R1-9", "R1-10", "R1-9"]
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
        assert resultado.vector_skus == ["A", "B", "FIN"]
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
        assert r.vector_skus == ["SKU_2", "SKU_3", "SKU_1"] and r.dic_descartes == {"sin ventas (unidades cero)": 1}
        r = extraer_secuencia(AdaptadorInventarioSintetico("WH_1"), ruta)
        assert r.vector_skus == ["SKU_2", "SKU_1"] and r.dic_descartes["otro almacén"] == 1


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
