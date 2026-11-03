"""Adaptador de calzado-furlan-de-assis-2024 (`Customer_Order.csv`): líneas de picking de una fábrica de calzado.

Todas las filas son salidas (líneas de pedido). Se descartan solo las de cantidad cero. La hora no es fiable (la fecha se registra por ola),
así que habrá muchos empates y se resuelven por el orden del archivo. No se descartan filas repetidas: cada línea es una recuperación.
"""
import csv
from datetime import datetime

from asrs.generacion.fuentes.base import Adaptador, Registro

ENCABEZADO = ["codCustomer", "orderNumber", "orderToCollect", "Reference", "Size (US)", "quantity (units)", "creationDate", "waveNumber", "operator"]


class AdaptadorCalzado(Adaptador):
    texto_nombre = "calzado"
    escalar_tope = 0.1       # la cabeza de la demanda queda individual y solo se agrupa la cola
    escalar_grupos = 180     # 208 referencias: se condensan a las 180 etiquetas del surtido del modelo sin quitar recuperaciones

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
        try:
            escalar_cantidad = float(fila["quantity (units)"])
        except ValueError:
            return Registro(txt_motivo="cantidad ilegible")
        if escalar_cantidad == 0:
            return Registro(txt_motivo="cantidad igual a cero")
        if not txt_referencia:
            return Registro(txt_motivo="referencia vacía")
        if self.bool_sku_con_talla:
            if not txt_talla:
                return Registro(txt_motivo="talla vacía (SKU con talla)")
            return Registro(f"{txt_referencia}-{txt_talla}", tiempo)
        return Registro(txt_referencia, tiempo)
