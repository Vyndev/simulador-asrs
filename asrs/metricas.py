"""Registro de tiempos y conteos de la simulación."""
import numpy as np
from statistics import stdev


class Metricas:

    def __init__(self):
        # Contadores generales para las operaciones de almacenamiento, recuperación y reubicacion
        self.cantidad_almacenamientos = 0
        self.cantidad_recuperaciones = 0
        self.cantidad_reubicaciones = 0
        self.cantidad_operaciones = 0

        # Contadores para las reubicaciones durante el almacenamiento y la recuperación
        self.cantidad_reubicaciones_almacenamiento = 0
        self.cantidad_reubicaciones_recuperacion = 0

        # Contadores para las operaciones de reubicación desagregadas
        self.cantidad_reubicaciones_regulares_almacenamiento = 0
        self.cantidad_reubicaciones_predeterminadas_almacenamiento = 0
        self.cantidad_reubicaciones_regulares_recuperacion = 0
        self.cantidad_reubicaciones_predeterminadas_recuperacion = 0

        # Contadores para el tiempo total de almacenamiento, recuperación y reubicacion
        self.tiempo_total_almacenamiento = 0
        self.tiempo_total_recuperacion = 0
        self.tiempo_total_reubicacion = 0

        # Contadores para el tiempo de operaciones de reubicacion durante el almacenamiento y la recuperación
        self.tiempo_total_reubicaciones_almacenamiento = 0
        self.tiempo_total_reubicaciones_recuperacion = 0
        self.tiempo_total_reubicaciones_regulares_almacenamiento = 0
        self.tiempo_total_reubicaciones_predeterminadas_almacenamiento = 0
        self.tiempo_total_reubicaciones_regulares_recuperacion = 0
        self.tiempo_total_reubicaciones_predeterminadas_recuperacion = 0
        self.tiempo_total_operacion = 0

        # Vectores de datos
        self.vector_tiempo_almacenamientos = []
        self.vector_tiempo_recuperaciones = []
        self.vector_tiempo_reubicaciones = []
        self.vector_tiempo_reubicaciones_almacenamiento = []
        self.vector_tiempo_reubicaciones_recuperacion = []
        self.vector_tiempo_reubicaciones_regulares_almacenamiento = []
        self.vector_tiempo_reubicaciones_predeterminadas_almacenamiento = []
        self.vector_tiempo_reubicaciones_regulares_recuperacion = []
        self.vector_tiempo_reubicaciones_predeterminadas_recuperacion = []

    def recuperarInformcion(self):
        self.cantidad_operaciones = self.cantidad_almacenamientos + \
            self.cantidad_recuperaciones
        self.tiempo_total_operacion = self.tiempo_total_almacenamiento + \
            self.tiempo_total_recuperacion + self.tiempo_total_reubicacion

        O0, O1 = self.cantidad_operaciones, self.tiempo_total_operacion
        A0, A1 = self.cantidad_almacenamientos, self.tiempo_total_almacenamiento
        B0, B1 = self.cantidad_recuperaciones, self.tiempo_total_recuperacion
        C0, C1 = self.cantidad_reubicaciones, self.tiempo_total_reubicacion
        D0, D1 = self.cantidad_reubicaciones_almacenamiento, self.tiempo_total_reubicaciones_almacenamiento
        E0, E1 = self.cantidad_reubicaciones_recuperacion, self.tiempo_total_reubicaciones_recuperacion
        F0, F1 = self.cantidad_reubicaciones_regulares_almacenamiento, self.tiempo_total_reubicaciones_regulares_almacenamiento
        G0, G1 = self.cantidad_reubicaciones_predeterminadas_almacenamiento, self.tiempo_total_reubicaciones_predeterminadas_almacenamiento
        H0, H1 = self.cantidad_reubicaciones_regulares_recuperacion, self.tiempo_total_reubicaciones_regulares_recuperacion
        I0, I1 = self.cantidad_reubicaciones_predeterminadas_recuperacion, self.tiempo_total_reubicaciones_predeterminadas_recuperacion

        A2 = stdev(self.vector_tiempo_almacenamientos) if self.vector_tiempo_almacenamientos and len(self.vector_tiempo_almacenamientos) > 1 else 0
        B2 = stdev(self.vector_tiempo_recuperaciones) if self.vector_tiempo_recuperaciones and len(self.vector_tiempo_recuperaciones) > 1 else 0
        C2 = stdev(self.vector_tiempo_reubicaciones) if self.vector_tiempo_reubicaciones and len(self.vector_tiempo_reubicaciones) > 1 else 0
        D2 = stdev(self.vector_tiempo_reubicaciones_almacenamiento) if self.vector_tiempo_reubicaciones_almacenamiento and len(self.vector_tiempo_reubicaciones_almacenamiento) > 1 else 0
        E2 = stdev(self.vector_tiempo_reubicaciones_recuperacion) if self.vector_tiempo_reubicaciones_recuperacion and len(self.vector_tiempo_reubicaciones_recuperacion) > 1 else 0
        F2 = stdev(self.vector_tiempo_reubicaciones_regulares_almacenamiento) if self.vector_tiempo_reubicaciones_regulares_almacenamiento and len(
self.vector_tiempo_reubicaciones_regulares_almacenamiento) > 1 else 0
        G2 = stdev(self.vector_tiempo_reubicaciones_predeterminadas_almacenamiento) if self.vector_tiempo_reubicaciones_predeterminadas_almacenamiento and len(self.vector_tiempo_reubicaciones_predeterminadas_almacenamiento) > 1 else 0
        H2 = stdev(
            self.vector_tiempo_reubicaciones_regulares_recuperacion) if self.vector_tiempo_reubicaciones_regulares_recuperacion and len(
            self.vector_tiempo_reubicaciones_regulares_recuperacion) > 1 else 0
        I2 = stdev(
            self.vector_tiempo_reubicaciones_predeterminadas_recuperacion) if self.vector_tiempo_reubicaciones_predeterminadas_recuperacion and len(
            self.vector_tiempo_reubicaciones_predeterminadas_recuperacion) > 1 else 0

        return [A0, A1, A2, B0, B1, B2, C0, C1, C2, D0, D1, D2, E0, E1, E2, F0, F1, F2, G0, G1, G2, H0, H1, H2, I0, I1, I2, O0, O1]

    def tiempoDezplamiento(self, vector_posicion_inicial, vector_posicion_final, vector_velocidad):
        vector_distancia = np.abs(
            vector_posicion_final - vector_posicion_inicial)

        # Determinar la distancia en el plano xy
        vector_distancia_xy = vector_distancia[:2]

        # Determina la distancia en el tramo z
        escalar_distancia_z = vector_distancia[2]

        # Calcula tiempos de Chebyshev el tramo xz
        vector_tiempo_xy = vector_distancia_xy / vector_velocidad[:2]
        esclar_tiempo_chebyshev_xy = np.max(vector_tiempo_xy)

        # Calcula tiempo de desplazamiento en z
        escalar_tiempo_z = escalar_distancia_z / vector_velocidad[2]

        # Calcula el tiempo total de ejecución
        tiempo_total_ejecucion = esclar_tiempo_chebyshev_xy + escalar_tiempo_z

        # Retorna el tiempo total de ejecución
        return tiempo_total_ejecucion

    def registrarAlmacenamiento(self, tiempo):
        # Agregar el tiempo de almacenamiento al DataFrame
        self.cantidad_almacenamientos += 1
        self.tiempo_total_almacenamiento += tiempo
        self.vector_tiempo_almacenamientos.append(tiempo)

    def registrarRecuperacion(self, tiempo):
        # Agregar el tiempo de recuperación al DataFrame
        self.cantidad_recuperaciones += 1
        self.tiempo_total_recuperacion += tiempo
        self.vector_tiempo_recuperaciones.append(tiempo)

    def registrarReubicacion(self, tiempo):
        # Agregar el tiempo de reubicación al DataFrame
        self.cantidad_reubicaciones += 1
        self.tiempo_total_reubicacion += tiempo
        self.vector_tiempo_reubicaciones.append(tiempo)

    def sumarTiempoReubicacion(self, tiempo):
        # Suma tiempo a la última reubicación registrada sin contarla de nuevo
        # Se usa para el retorno de ítems desde el puntoES, que ocurre después de registrar la reubicación
        self.tiempo_total_reubicacion += tiempo
        if self.vector_tiempo_reubicaciones:
            self.vector_tiempo_reubicaciones[-1] += tiempo

    def registrarReubicacionAlmacenamiento(self, tiempo):
        # Agregar el tiempo de reubicación durante el almacenamiento al DataFrame
        self.cantidad_reubicaciones_almacenamiento += 1
        self.tiempo_total_reubicaciones_almacenamiento += tiempo
        self.vector_tiempo_reubicaciones_almacenamiento.append(tiempo)

    def registrarReubicacionRecuperacion(self, tiempo):
        # Agregar el tiempo de reubicación durante la recuperación al DataFrame
        self.cantidad_reubicaciones_recuperacion += 1
        self.tiempo_total_reubicaciones_recuperacion += tiempo
        self.vector_tiempo_reubicaciones_recuperacion.append(tiempo)

    def registrarReubicacionRegularAlmacenamiento(self, tiempo):
        # Agregar el tiempo de reubicación regular durante el almacenamiento al DataFrame
        self.cantidad_reubicaciones_regulares_almacenamiento += 1
        self.tiempo_total_reubicaciones_regulares_almacenamiento += tiempo
        self.vector_tiempo_reubicaciones_regulares_almacenamiento.append(
            tiempo)

    def registrarReubicacionPredeterminadaAlmacenamiento(self, tiempo):
        # Agregar el tiempo de reubicación predeterminada durante el almacenamiento al DataFrame
        self.cantidad_reubicaciones_predeterminadas_almacenamiento += 1
        self.tiempo_total_reubicaciones_predeterminadas_almacenamiento += tiempo
        self.vector_tiempo_reubicaciones_predeterminadas_almacenamiento.append(
            tiempo)

    def registrarReubicacionRegularRecuperacion(self, tiempo):
        # Agregar el tiempo de reubicación regular durante la recuperación al DataFrame
        self.cantidad_reubicaciones_regulares_recuperacion += 1
        self.tiempo_total_reubicaciones_regulares_recuperacion += tiempo
        self.vector_tiempo_reubicaciones_regulares_recuperacion.append(tiempo)

    def registrarReubicacionPredeterminadaRecuperacion(self, tiempo):
        # Agregar el tiempo de reubicación predeterminada durante la recuperación al DataFrame
        self.cantidad_reubicaciones_predeterminadas_recuperacion += 1
        self.tiempo_total_reubicaciones_predeterminadas_recuperacion += tiempo
        self.vector_tiempo_reubicaciones_predeterminadas_recuperacion.append(
            tiempo)
