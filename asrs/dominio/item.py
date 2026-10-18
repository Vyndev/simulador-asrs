"""Ítem almacenable."""


class Item:
    escalar_contador_global_generacion = 0

    # Se definen los atributos constructores
    def __init__(self, escalar_etiqueta_clase) -> None:
        # Se aumenta en 1 el contador de ítems generados
        Item.escalar_contador_global_generacion += 1

        # Se guarda la etiqueta de generación
        self.escalar_etiqueta_generacion = Item.escalar_contador_global_generacion

        # Se guarda la etiqueta de clase
        self.escalar_etiqueta_clase = escalar_etiqueta_clase

    # Se configura la forma en que se representa la instancia en el momento que se llama
    def __repr__(self) -> str:
        return f'{self.escalar_etiqueta_clase}|{self.escalar_etiqueta_generacion}'
