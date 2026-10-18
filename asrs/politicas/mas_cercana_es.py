"""Política de la posición libre más cercana al punto E/S."""
from asrs.politicas.asignacion import PoliticaAsignacion


class PoliticaMasCercanaAlES(PoliticaAsignacion):
    nombre = 'MasCercanaAlES'

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Tiempo de viaje Chebyshev desde el punto E/S, la referencia de la operación no cambia la política
        return self.tiempoChebyshevXY(self.instancia_puntoES.vector_posicion)
