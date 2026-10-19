"""Graba la ejecución de las políticas y genera los datos que reproduce el visualizador."""
import json
import os
import random as rnd

import numpy as np

from asrs.control.controlador_grua import ControladorGrua
from asrs.politicas.zonificacion import clasificar_skus
from asrs.simulacion.simulacion import Simulacion


def _rango_por_demanda(vector_demanda):
    # Ranking de rotación: 0 para el SKU más demandado, a igual demanda se conserva el orden por número de SKU
    vector_rango = np.empty(len(vector_demanda), dtype=int)
    vector_rango[np.argsort(-np.asarray(vector_demanda), kind='stable')] = np.arange(len(vector_demanda))
    return vector_rango


def grabar_escenario(instancia_simulacion, txt_clave_politica, SA, S):
    # Ejecuta una política registrando todos los eventos de la grúa durante la secuencia de operación
    E, G, P, instancia_politica, instancia_politica_recuperacion = instancia_simulacion.prepararEscenario(txt_clave_politica, SA)

    # Estado inicial de la estantería, un SKU por celda (0 = vacía) en orden fila, columna, capa
    vector_estado_inicial = [int(instancia_item.escalar_sku) if instancia_item != 0 else 0
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
        # Cómo se colorean los ítems: la política basada en clases usa la clase de cada SKU y las demás la rotación del SKU
        "color_por": "clase" if txt_clave_politica == 'basada_en_clases' else "sku",
        "inicial": vector_estado_inicial,
        "eventos": G.vector_eventos,
        "tiempo_total": round(float(vector_metricas[-1]), 4),
        "reubicaciones": int(vector_metricas[6]),
        "reloj_final": round(float(G.escalar_reloj), 4),
    }


def generar_datos(config, vector_semillas, al_avanzar=None):
    # Graba cada política en cada réplica, usando las mismas semillas que la simulación principal
    # al_avanzar: función opcional que se llama al terminar cada réplica, sirve para informar el progreso
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
        if al_avanzar is not None:
            al_avanzar()

    return {
        "configuracion": [int(escalar) for escalar in config["vector_configuracion_estanteria"]],
        "cotas": [float(escalar) for escalar in config["vector_cotas_estanteria"]],
        "velocidad": [float(escalar) for escalar in config["vector_velocidad_grua"]],
        "aceleracion": [float(escalar) for escalar in config["vector_aceleracion_grua"]],
        "deceleracion": [float(escalar) for escalar in config["vector_deceleracion_grua"]],
        "posicion": [float(escalar) for escalar in config["vector_posicion_estanteria"]],
        "centro": [float(escalar) for escalar in config["vector_centro_celda_estanteria"]],
        "skus": int(config["escalar_numero_skus"]),
        "demanda_sku": [round(float(escalar), 6) for escalar in config["vector_demanda_sku"]],
        # Clase de cada SKU según la agrupación de la política basada en clases, solo para colorear la animación
        "clase_sku": [int(escalar) for escalar in clasificar_skus(config["vector_demanda_sku"], config)],
        # Posición de cada SKU en el ranking de rotación (0 = el más demandado), indexada por SKU - 1
        "rango_sku": [int(escalar) for escalar in _rango_por_demanda(config["vector_demanda_sku"])],
        # Etiquetas originales de los SKUs (solo con datos reales): el SKU número i se muestra con su código
        "etiquetas_sku": list(config.get("etiquetas_sku", [])) if config.get("texto_origen_tareas") == "datos" else [],
        "z_riel": float(instancia_simulacion.crearInstancias()[1].vector_posicion_resposo[2]),
        "replicas": vector_replicas,
    }


def leer_plantilla():
    # Devuelve el HTML de la página con el marcador de datos sin reemplazar
    txt_ruta_plantilla = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plantilla.html")
    with open(txt_ruta_plantilla, encoding="utf-8") as archivo:
        txt_pagina = archivo.read()
    # La librería gráfica (Three.js) se incluye en la página para que funcione sin conexión a internet
    txt_ruta_libreria = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "three.min.js")
    with open(txt_ruta_libreria, encoding="utf-8") as archivo:
        return txt_pagina.replace("<!--THREE-->", "<script>" + archivo.read() + "</script>")


def escribir_html(diccionario_datos, txt_ruta_salida):
    # Inserta los datos en la plantilla y escribe la página, que funciona sin servidor
    txt_plantilla = leer_plantilla()
    txt_datos = json.dumps(diccionario_datos, separators=(",", ":"))
    with open(txt_ruta_salida, "w", encoding="utf-8", newline="\n") as archivo:
        archivo.write(txt_plantilla.replace("/*DATOS*/null", txt_datos))
