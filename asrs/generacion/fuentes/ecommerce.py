"""Adaptador de ecommerce-roy-bansal-2020 (`E-Commerce DataSet.xlsx`): líneas de pedido de un minorista de comercio electrónico.

Todas las filas son salidas; una línea de cantidad q (`prod_qty`) son q recuperaciones seguidas del SKU. La demanda se ordena por la fecha del pedido (`order_date`); la de envío no interviene.
El surtido se agrupa en 180 etiquetas con k-means (ver `agrupar_surtido`). Se descartan las fechas fuera de la campaña (ventana configurable, por defecto agosto a diciembre de 2015), las cantidades cero y los SKU vacíos.
El archivo es un .xlsx y se lee con la biblioteca estándar (zip + XML), sin openpyxl.
"""
import datetime
import re
import xml.etree.ElementTree as ET
import zipfile

from asrs.generacion.fuentes.base import Adaptador, Registro, leer_cantidad

ENCABEZADO = ["rec_id", "order_id", "order_date", "shipped_at", "prod_sku", "prod_qty"]
ORIGEN_EXCEL = datetime.datetime(1899, 12, 30)
ESPACIO = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _indice_columna(txt_referencia):
    escalar = 0
    for txt_letra in re.match(r"[A-Z]+", txt_referencia).group(0):
        escalar = escalar * 26 + ord(txt_letra) - 64
    return escalar - 1


def leer_hoja_xlsx(txt_ruta):
    """Itera las filas de la primera hoja de un .xlsx como listas de texto (las celdas vacías son '')."""
    with zipfile.ZipFile(txt_ruta) as zip_xlsx:
        vector_textos = []
        for _, elemento in ET.iterparse(zip_xlsx.open("xl/sharedStrings.xml")):
            if elemento.tag.endswith("}si"):
                vector_textos.append("".join(t.text or "" for t in elemento.iter() if t.tag.endswith("}t")))
                elemento.clear()
        for _, elemento in ET.iterparse(zip_xlsx.open("xl/worksheets/sheet1.xml")):
            if elemento.tag.endswith("}row"):
                dic_celdas = {}
                for celda in elemento.findall("m:c", ESPACIO):
                    valor = celda.find("m:v", ESPACIO)
                    if valor is not None:
                        dic_celdas[_indice_columna(celda.get("r"))] = vector_textos[int(valor.text)] if celda.get("t") == "s" else valor.text
                if dic_celdas:
                    yield [dic_celdas.get(i, "") for i in range(max(dic_celdas) + 1)]
                elemento.clear()


def _fecha_excel(txt_valor):
    # Días desde 1899-12-30, redondeados al segundo para que dos celdas del mismo instante no difieran por el error de coma flotante
    return ORIGEN_EXCEL + datetime.timedelta(seconds=round(float(txt_valor) * 86400))


class AdaptadorEcommerce(Adaptador):
    # Miles de SKU no caben en la estantería: se agrupan en 180 etiquetas con k-means (surtido de 180 SKU del modelo) sin quitar recuperaciones
    escalar_grupos = 180

    def __init__(self, txt_fecha_minima="2015-08-01", txt_fecha_maxima="2016-01-01"):
        # Ventana [mínima, máxima) de fechas de pedido aceptadas, en ISO. Según calidad.md la campaña va de agosto a diciembre de 2015
        self.fecha_minima = datetime.datetime.fromisoformat(txt_fecha_minima)
        self.fecha_maxima = datetime.datetime.fromisoformat(txt_fecha_maxima)

    def filas(self, txt_ruta):
        iterador = leer_hoja_xlsx(txt_ruta)
        vector_encabezado = next(iterador)
        if vector_encabezado != ENCABEZADO:
            raise ValueError(f"Ecommerce: encabezado inesperado {vector_encabezado}.")
        for vector_fila in iterador:
            yield dict(zip(ENCABEZADO, vector_fila + [""] * (len(ENCABEZADO) - len(vector_fila))))

    def interpretar(self, fila):
        txt_fecha = fila["order_date"].strip()
        if txt_fecha in ("", "NA"):
            return Registro(txt_motivo="fecha de pedido ausente")
        try:
            tiempo = _fecha_excel(txt_fecha)
        except ValueError:
            return Registro(txt_motivo="fecha de pedido ilegible")
        if not self.fecha_minima <= tiempo < self.fecha_maxima:
            return Registro(txt_motivo="fecha de pedido fuera de la campaña")
        escalar_cantidad, txt_motivo = leer_cantidad(fila["prod_qty"])
        if txt_motivo:
            return Registro(txt_motivo=txt_motivo)
        txt_sku = fila["prod_sku"].strip()
        if not txt_sku:
            return Registro(txt_motivo="SKU vacío")
        return Registro(txt_sku, tiempo, escalar_cantidad=escalar_cantidad)
