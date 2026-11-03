"""Registro de adaptadores: nombre de la fuente -> (carpeta en tesis/datos, archivo crudo, clase del adaptador)."""
from asrs.generacion.fuentes.calzado import AdaptadorCalzado
from asrs.generacion.fuentes.ecommerce import AdaptadorEcommerce
from asrs.generacion.fuentes.inventario_sintetico import AdaptadorInventarioSintetico

# Para añadir una fuente: escribir su adaptador y añadir una línea aquí
ADAPTADORES = {
    "calzado": ("calzado-furlan-de-assis-2024", "Customer_Order.csv", AdaptadorCalzado),
    "ecommerce": ("ecommerce-roy-bansal-2020", "E-Commerce DataSet.xlsx", AdaptadorEcommerce),
    "inventario_sintetico": ("inventario-sintetico-ziya-2025", "high-dimensional-supply-chain-inventory-dataset.zip", AdaptadorInventarioSintetico),
}


def crear_adaptador(txt_nombre, **opciones):
    if txt_nombre not in ADAPTADORES:
        raise ValueError(f"Fuente desconocida '{txt_nombre}'. Disponibles: {', '.join(ADAPTADORES)}.")
    return ADAPTADORES[txt_nombre][2](**opciones)
