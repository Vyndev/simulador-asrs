"""Graba la ejecución de las políticas y genera una página HTML que las compara en carrera."""
import json
import os
import random as rnd

import numpy as np

from asrs.control.controlador_grua import ControladorGrua
from asrs.simulacion.simulacion import Simulacion


def grabar_escenario(instancia_simulacion, txt_clave_politica, SA, S):
    # Ejecuta una política registrando todos los eventos de la grúa durante la secuencia de operación
    E, G, P, instancia_politica, instancia_politica_recuperacion = instancia_simulacion.prepararEscenario(txt_clave_politica, SA)

    # Estado inicial de la estantería, una etiqueta de clase por celda (0 = vacía) en orden fila, columna, capa
    vector_estado_inicial = [int(instancia_item.escalar_etiqueta_clase) if instancia_item != 0 else 0
                             for instancia_item in E.matriz_contenedor_items.flat]

    # Desde aquí el reloj de la grúa empieza en cero y se registran los eventos
    G.escalar_reloj = 0.0
    G.vector_eventos = []
    C = ControladorGrua(E, G, P, S, instancia_politica, instancia_politica_recuperacion)
    C.ejecutarSecuencia()
    G.registrarEvento('F')

    vector_metricas = C.instancia_metricas.recuperarInformcion()
    return {
        "nombre": instancia_politica.nombre,
        "inicial": vector_estado_inicial,
        "eventos": G.vector_eventos,
        "tiempo_total": round(float(vector_metricas[-1]), 4),
        "reubicaciones": int(vector_metricas[6]),
        "reloj_final": round(float(G.escalar_reloj), 4),
    }


def generar_datos(config, vector_semillas):
    # Graba cada política en cada réplica, usando las mismas semillas que la simulación principal
    instancia_simulacion = Simulacion(config)
    vector_replicas = []

    for escalar_semilla in vector_semillas:
        rnd.seed(escalar_semilla)
        SA, S = instancia_simulacion.generarSecuencias()

        vector_politicas = []
        for escalar_indice_politica, txt_clave_politica in enumerate(config["vector_politicas_asignacion"]):
            rnd.seed(escalar_semilla * 1000 + escalar_indice_politica)
            vector_politicas.append(grabar_escenario(instancia_simulacion, txt_clave_politica, SA, S))

        vector_replicas.append({
            "semilla": int(escalar_semilla),
            "secuencia": [[int(escalar_actividad), int(escalar_clase)] for escalar_actividad, escalar_clase in S],
            "politicas": vector_politicas,
        })

    return {
        "configuracion": [int(escalar) for escalar in config["vector_configuracion_estanteria"]],
        "cotas": [float(escalar) for escalar in config["vector_cotas_estanteria"]],
        "velocidad": [float(escalar) for escalar in config["vector_velocidad_grua"]],
        "clases": len(config["vector_demanda_clase"]),
        "z_riel": float(instancia_simulacion.crearInstancias()[1].vector_posicion_resposo[2]),
        "replicas": vector_replicas,
    }


def escribir_html(diccionario_datos, txt_ruta_salida):
    # Inserta los datos en la plantilla y escribe la página, que funciona sin servidor
    txt_ruta_plantilla = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plantilla.html")
    with open(txt_ruta_plantilla, encoding="utf-8") as archivo:
        txt_plantilla = archivo.read()

    txt_datos = json.dumps(diccionario_datos, separators=(",", ":"))
    with open(txt_ruta_salida, "w", encoding="utf-8", newline="\n") as archivo:
        archivo.write(txt_plantilla.replace("/*DATOS*/null", txt_datos))
