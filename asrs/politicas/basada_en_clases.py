"""Política basada en clases: los SKUs se agrupan en clases y cada clase tiene una zona fija."""
import numpy as np

from asrs.generacion.probabilidades import perfil_inventario
from asrs.politicas.asignacion import PoliticaAsignacion
from asrs.politicas.zonificacion import clasificar_skus, construir_zonificacion


class PoliticaBasadaEnClases(PoliticaAsignacion):
    nombre = 'BasadaEnClases'

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, config):
        super().__init__(instancia_estanteria, instancia_grua, instancia_puntoES, config)

        # El plan se forma antes del llenado con la popularidad móvil del día 0 y rige también durante el llenado
        self.planificar(self.demandaPlanificacion(0))

    def planificar(self, vector_demanda):
        # La agrupación es parte de la política: los SKUs más demandados forman la clase 1 y reciben las celdas más cercanas al punto E/S
        # Las unidades ya guardadas no se mueven, las zonas nuevas rigen para los almacenamientos siguientes
        self.vector_clase_de_sku = clasificar_skus(vector_demanda, self.config)

        # El espacio de cada clase es proporcional al inventario de sus SKUs (según el perfil de inventario), no a su demanda
        escalar_clases = int(self.vector_clase_de_sku.max())
        vector_inventario = perfil_inventario(vector_demanda, self.config["texto_perfil_inventario"])
        vector_espacio = np.array([vector_inventario[self.vector_clase_de_sku == clase].sum() for clase in range(1, escalar_clases + 1)])
        self.matriz_zonificacion = construir_zonificacion(
            self.tiempoChebyshevXY(self.instancia_puntoES.vector_posicion),
            vector_espacio,
            list(range(1, escalar_clases + 1)))

    def etiquetaZona(self, escalar_sku):
        return int(self.vector_clase_de_sku[escalar_sku - 1])

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Dentro de la zona de la clase se prefiere la celda más cercana al punto E/S
        return self.tiempoChebyshevXY(self.instancia_puntoES.vector_posicion)


class PoliticaBasadaEnClasesReplanificada(PoliticaBasadaEnClases):
    nombre = 'BasadaEnClasesReplanificada'

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, config):
        super().__init__(instancia_estanteria, instancia_grua, instancia_puntoES, config)
        # Último periodo de replanificación en que se rehizo el plan, el 0 es el plan inicial
        self.escalar_ultimo_periodo_replanificado = 0

    def iniciarDia(self, escalar_dia):
        # Parte del plan inicial y rehace las clases en el primer día de cada periodo nuevo (el día 0 ya tiene el plan inicial)
        # Se compara el periodo del día con el último replanificado y no se exige un múltiplo exacto, así un día omitido no salta una replanificación
        if self.config.get("escenario_demanda") is None or escalar_dia <= 0:
            return
        escalar_periodo = escalar_dia // self.config["escalar_dias_replanificacion"]
        if escalar_periodo > self.escalar_ultimo_periodo_replanificado:
            self.escalar_ultimo_periodo_replanificado = escalar_periodo
            self.planificar(self.demandaPlanificacion(escalar_dia))
