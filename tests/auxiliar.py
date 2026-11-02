"""Ayudas comunes de las pruebas: deja la raíz del proyecto en sys.path y prepara configuraciones y estados de estantería."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asrs.config import cargar_configuracion as _cargar_configuracion_instancia
from asrs.dominio.item import Item
from asrs.politicas.registro import crear_politica
from asrs.simulacion.simulacion import Simulacion as _Simulacion

# Las pruebas usan los parámetros de la instancia del capítulo 6, pero con una lista de operación corta (el 10 % de las celdas)
# para que se ejecuten en segundos: la instancia completa evalúa un año de tareas (525 600) por corrida
TAREAS_PRUEBA = 72


def cargar_configuracion():
    config = _cargar_configuracion_instancia()
    config["escalar_largo_secuencia"] = TAREAS_PRUEBA
    return config


class Simulacion(_Simulacion):
    def __init__(self, config=None):
        super().__init__(config if config is not None else cargar_configuracion())


def preparar_estado(txt_clave_politica):
    # Estantería llena al nivel configurado bajo la política dada
    simulacion = Simulacion()
    SA, S = simulacion.generarSecuencias()
    random.seed(1)
    from asrs.control.controlador_grua import ControladorGrua
    Item.escalar_contador_global_generacion = 0
    E, G, P = simulacion.crearInstancias()
    politica = crear_politica(txt_clave_politica, E, G, P, simulacion.configPolitica())
    ControladorGrua(E, G, P, SA, politica).ejecutarSecuencia()
    return simulacion, E, G, P, politica
