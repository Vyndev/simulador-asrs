"""Política full-turnover (Hausman, Graves y Schwarz, 1976): cada SKU tiene su propia zona fija."""
import numpy as np

from asrs.generacion.probabilidades import perfil_inventario
from asrs.politicas.asignacion import PoliticaAsignacion
from asrs.politicas.zonificacion import construir_zonificacion


class PoliticaFullTurnover(PoliticaAsignacion):
    nombre = 'FullTurnover'

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, config):
        super().__init__(instancia_estanteria, instancia_grua, instancia_puntoES, config)

        # El plan se forma antes del llenado con la popularidad móvil del día 0 y rige también durante el llenado
        self.planificar(self.demandaPlanificacion(0))

    def planificar(self, vector_demanda):
        # Los SKUs se ordenan de mayor a menor demanda y el más demandado recibe las ubicaciones fijas más cercanas
        # A igual demanda se conserva el orden por número de SKU. El espacio de cada SKU es proporcional a su inventario
        # Con el perfil EOQ el orden por demanda coincide con el orden por rotación (demanda / inventario = raíz de la demanda)
        # Las unidades ya guardadas no se mueven, las zonas nuevas rigen para los almacenamientos siguientes
        vector_orden_skus = [int(escalar_indice) + 1 for escalar_indice in np.argsort(-vector_demanda, kind='stable')]
        self.matriz_zonificacion = construir_zonificacion(
            self.tiempoChebyshevXY(self.instancia_puntoES.vector_posicion),
            perfil_inventario(vector_demanda, self.config["texto_perfil_inventario"]),
            vector_orden_skus)

    def etiquetaZona(self, escalar_sku):
        # Cada SKU es su propia zona
        return int(escalar_sku)

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Dentro de la zona del SKU se prefiere la celda más cercana al punto E/S
        return self.tiempoChebyshevXY(self.instancia_puntoES.vector_posicion)


class PoliticaFullTurnoverReplanificada(PoliticaFullTurnover):
    nombre = 'FullTurnoverReplanificada'

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, config):
        super().__init__(instancia_estanteria, instancia_grua, instancia_puntoES, config)
        # Último periodo de replanificación en que se rehizo el plan, el 0 es el plan inicial
        self.escalar_ultimo_periodo_replanificado = 0

    def iniciarDia(self, escalar_dia):
        # Parte del plan inicial y rehace las zonas en el primer día de cada periodo nuevo (el día 0 ya tiene el plan inicial)
        # Se compara el periodo del día con el último replanificado y no se exige un múltiplo exacto, así un día omitido no salta una replanificación
        if self.config.get("escenario_demanda") is None or escalar_dia <= 0:
            return
        escalar_periodo = escalar_dia // self.config["escalar_dias_replanificacion"]
        if escalar_periodo > self.escalar_ultimo_periodo_replanificado:
            self.escalar_ultimo_periodo_replanificado = escalar_periodo
            self.planificar(self.demandaPlanificacion(escalar_dia))
