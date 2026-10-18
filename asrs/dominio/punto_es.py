"""Punto de entrada y salida."""


class PuntoES:
    def __init__(self, vector_posicion) -> None:
        # Inicializa el punto E/S con una posición dada
        # La posición se guarda en 'vector_posicion'
        self.vector_posicion = vector_posicion

        # Inicializa un contenedor temporal y de salida para los ítems
        # Inicialmente los contenedores son listas vacías
        self.matriz_contenedor_temporal_items = []
        self.matriz_contenedor_salida_items = []

    def __repr__(self) -> str:
        # Este método devuelve una representación en cadena de caracteres del objeto
        # La cadena incluye la posición del punto E/S y los ítems en el contenedor temporal
        return f'{self.vector_posicion}|{self.matriz_contenedor_temporal_items}'
