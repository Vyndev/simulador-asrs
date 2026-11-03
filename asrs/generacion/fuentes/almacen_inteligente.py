"""Adaptador de almacen-inteligente-colabsss-2026 (`iot_warehouse_optimization_dataset.csv` dentro de un .zip): lecturas de un almacén. DATOS ALEATORIOS.

Cada fila es una lectura de un producto en una estantería. Es demanda cuando la lectura registra artículos que salen (`outgoing_items` > 0): esa fila
cuenta como una recuperación del producto. Se ordena por `timestamp`. Las entradas (`incoming_items`) no se usan. Según calidad.md los valores parecen
sorteados al azar, con 600 productos y demanda casi uniforme: sirve para probar el modelo, no como evidencia de realismo.
"""
from datetime import datetime

from asrs.generacion.fuentes.base import Adaptador, Registro, filas_csv_en_zip

COLUMNAS_NECESARIAS = ["timestamp", "product_id", "outgoing_items"]


class AdaptadorAlmacenInteligente(Adaptador):
    texto_nombre = "almacen_inteligente"
    txt_archivo_interno = "iot_warehouse_optimization_dataset.csv"
    escalar_tope = 1.5       # con 0,1 casi todos los SKU (demanda casi uniforme) superarían el umbral y no quedaría cupo para la cola
    escalar_grupos = 180     # 600 productos no caben en el surtido del modelo: se agrupan en 180 etiquetas sin quitar recuperaciones

    def filas(self, txt_ruta):
        for escalar_numero, fila in enumerate(filas_csv_en_zip(txt_ruta, self.txt_archivo_interno)):
            if escalar_numero == 0 and any(txt not in fila for txt in COLUMNAS_NECESARIAS):
                raise ValueError(f"Almacén inteligente: faltan columnas {COLUMNAS_NECESARIAS} en {list(fila)}.")
            yield fila

    def interpretar(self, fila):
        try:
            tiempo = datetime.strptime(fila["timestamp"].strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return Registro(txt_motivo="fecha ilegible")
        try:
            escalar_salidas = float(fila["outgoing_items"])
        except ValueError:
            return Registro(txt_motivo="salidas ilegibles")
        if escalar_salidas <= 0:
            return Registro(txt_motivo="sin salidas (outgoing_items cero)")
        txt_sku = fila["product_id"].strip()
        if not txt_sku:
            return Registro(txt_motivo="producto vacío")
        return Registro(txt_sku, tiempo)
