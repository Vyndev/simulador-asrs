"""Ayudas comunes de las pruebas: deja la raíz del proyecto en sys.path y prepara estados de estantería."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asrs.dominio.item import Item
from asrs.politicas.registro import crear_politica
from asrs.simulacion.simulacion import Simulacion


def preparar_estado(txt_clave_politica):
    # Estantería llena al nivel configurado bajo la política dada
    simulacion = Simulacion()
    SA, S = simulacion.generarSecuencias()
    random.seed(1)
    from asrs.control.controlador_grua import ControladorGrua
    Item.escalar_contador_global_generacion = 0
    E, G, P = simulacion.crearInstancias()
    politica = crear_politica(txt_clave_politica, E, G, P, simulacion.config)
    ControladorGrua(E, G, P, SA, politica).ejecutarSecuencia()
    return simulacion, E, G, P, politica
