"""Política de la posición libre más cercana a la grúa."""
from asrs.politicas.asignacion import PoliticaAsignacion


class PoliticaMasCercanaALaGrua(PoliticaAsignacion):
    nombre = 'MasCercanaALaGrua'

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Tiempo de viaje Chebyshev desde la grúa
        # Al almacenar la grúa está en reposo (en el punto E/S)
        # Al reubicar la grúa estará en la celda del ítem que retira, esa posición llega como referencia
        if vector_posicion_referencia is None:
            vector_posicion_referencia = self.instancia_grua.vector_posicion_actual
        return self.tiempoChebyshevXY(vector_posicion_referencia)
