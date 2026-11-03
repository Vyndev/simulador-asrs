"""Adaptador de inventario-sintetico-ziya-2025 (`supply_chain_dataset1.csv` dentro de un .zip): ventas diarias de 50 SKU. DATOS SIMULADOS.

Cada fila es la venta de un día de un SKU en un almacén (hay cinco almacenes por SKU y día), no una línea de pedido. Se toma cada fila con ventas
como una recuperación, igual que una línea de pedido en las otras fuentes, y se ordena por fecha. La resolución es el día: dentro de un día
se conserva el orden del archivo, que va por SKU y luego por fecha. Se descartan las filas sin ventas (unidades cero). Opcionalmente se puede
dejar un solo almacén. Según calidad.md todos los SKU tienen la misma demanda: sirve para probar el modelo, no como evidencia de realismo.
"""
from datetime import datetime

from asrs.generacion.fuentes.base import Adaptador, Registro, filas_csv_en_zip

ENCABEZADO = ["Date", "SKU_ID", "Warehouse_ID", "Supplier_ID", "Region", "Units_Sold", "Inventory_Level", "Supplier_Lead_Time_Days", "Reorder_Point",
              "Order_Quantity", "Unit_Cost", "Unit_Price", "Promotion_Flag", "Stockout_Flag", "Demand_Forecast"]


class AdaptadorInventarioSintetico(Adaptador):
    txt_archivo_interno = "supply_chain_dataset1.csv"

    def __init__(self, txt_almacen=None):
        self.txt_almacen = txt_almacen      # p. ej. "WH_1" para tratar un solo almacén; None conserva los cinco

    def filas(self, txt_ruta):
        for escalar_numero, fila in enumerate(filas_csv_en_zip(txt_ruta, self.txt_archivo_interno)):
            if escalar_numero == 0 and list(fila) != ENCABEZADO:
                raise ValueError(f"Inventario sintético: encabezado inesperado {list(fila)}.")
            yield fila

    def interpretar(self, fila):
        if self.txt_almacen is not None and fila["Warehouse_ID"].strip() != self.txt_almacen:
            return Registro(txt_motivo="otro almacén")
        try:
            tiempo = datetime.strptime(fila["Date"].strip(), "%Y-%m-%d")
        except ValueError:
            return Registro(txt_motivo="fecha ilegible")
        try:
            escalar_unidades = float(fila["Units_Sold"])
        except ValueError:
            return Registro(txt_motivo="unidades ilegibles")
        if escalar_unidades <= 0:
            return Registro(txt_motivo="sin ventas (unidades cero)")
        txt_sku = fila["SKU_ID"].strip()
        if not txt_sku:
            return Registro(txt_motivo="SKU vacío")
        return Registro(txt_sku, tiempo)
