"""Adaptador de calzado-furlan-de-assis-2024 (`Customer_Order.csv`): líneas de picking de una fábrica de calzado.

Todas las filas son salidas (líneas de pedido). Se descartan solo las de cantidad cero. La hora no es fiable (la fecha se registra por ola),
así que habrá muchos empates y se resuelven por el orden del archivo. No se descartan filas repetidas. Una línea de cantidad q son q recuperaciones seguidas de la referencia (quantity (units)).
"""
import csv
from datetime import datetime

from asrs.generacion.fuentes.base import Adaptador, Registro, leer_cantidad

ENCABEZADO = ["codCustomer", "orderNumber", "orderToCollect", "Reference", "Size (US)", "quantity (units)", "creationDate", "waveNumber", "operator"]


class AdaptadorCalzado(Adaptador):
    escalar_max_skus = 180   # 208 referencias: el surtido se reduce a 180 SKU con muestreo ponderado por la demanda

    def __init__(self, bool_sku_con_talla=False):
        # Por defecto el SKU es la referencia (208 SKU); con talla es referencia-talla (2517 SKU, como `preparar.py`)
        self.bool_sku_con_talla = bool_sku_con_talla

    def filas(self, txt_ruta):
        with open(txt_ruta, encoding="utf-8-sig", newline="") as archivo:
            lector = csv.reader(archivo, delimiter=";")
            vector_encabezado = [txt.strip() for txt in next(lector)]
            if vector_encabezado != ENCABEZADO:
                raise ValueError(f"Calzado: encabezado inesperado {vector_encabezado}.")
            for vector_fila in lector:
                yield dict(zip(ENCABEZADO, vector_fila))

    def interpretar(self, fila):
        try:
            tiempo = datetime.strptime(fila["creationDate"].strip(), "%d/%m/%Y %H:%M")
        except ValueError:
            return Registro(txt_motivo="fecha de creación ilegible")
        txt_referencia, txt_talla = fila["Reference"].strip(), fila["Size (US)"].strip()
        escalar_cantidad, txt_motivo = leer_cantidad(fila["quantity (units)"])
        if txt_motivo:
            return Registro(txt_motivo=txt_motivo)
        if not txt_referencia:
            return Registro(txt_motivo="referencia vacía")
        if self.bool_sku_con_talla:
            if not txt_talla:
                return Registro(txt_motivo="talla vacía (SKU con talla)")
            return Registro(f"{txt_referencia}-{txt_talla}", tiempo, escalar_cantidad=escalar_cantidad)
        return Registro(txt_referencia, tiempo, escalar_cantidad=escalar_cantidad)
