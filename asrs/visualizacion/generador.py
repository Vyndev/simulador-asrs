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


def vectores_por_sku(instancia_simulacion, config):
    # Demanda, clase y ranking de rotación de cada SKU que usa la animación para colorear, indexados por SKU - 1
    # Con un escenario generado el SKU es el número de su posición del surtido y la altura de su curva es aleatoria: lo que las políticas ven
    # es la popularidad móvil del día 0, así que se ordena y se agrupa con ella. Con datos reales se conserva la demanda de la configuración
    vector_demanda = np.asarray(config["vector_demanda_sku"], dtype=float)
    instancia_escenario = instancia_simulacion.escenario_demanda
    if instancia_escenario is not None:
        vector_popularidad = np.asarray(instancia_escenario.popularidad(0), dtype=float)
        # Si no hay recuperaciones en la ventana no se puede repartir y se usa la demanda de la configuración, como hacen las políticas
        if vector_popularidad.sum() > 0:
            vector_demanda = vector_popularidad / vector_popularidad.sum()
    return {
        "demanda_sku": [round(float(escalar), 6) for escalar in vector_demanda],
        # Clase de cada SKU según la agrupación de la política basada en clases, solo para colorear la animación
        "clase_sku": [int(escalar) for escalar in clasificar_skus(vector_demanda, config)],
        # Posición de cada SKU en el ranking de rotación (0 = el más demandado)
        "rango_sku": [int(escalar) for escalar in _rango_por_demanda(vector_demanda)],
    }


def grabar_escenario(instancia_simulacion, txt_clave_politica, SA, S, txt_clave_recuperacion = 'fcfs'):
    # Ejecuta una política registrando todos los eventos de la grúa durante la secuencia de operación
    E, G, P, instancia_politica, instancia_politica_recuperacion = instancia_simulacion.prepararEscenario(txt_clave_politica, SA, txt_clave_recuperacion)

    # Estado inicial de la estantería, un SKU por celda (0 = vacía) en orden fila, columna, capa
    vector_estado_inicial = [int(instancia_item.escalar_sku) if instancia_item != 0 else 0
                             for instancia_item in E.matriz_contenedor_items.flat]

    # Desde aquí el reloj de la grúa empieza en cero y se registran los eventos
    G.escalar_reloj = 0.0
    G.vector_eventos = []
    # Con escenario de demanda, el controlador recibe el día de cada tarea y los retiros de las tareas que se graban (las de S)
    C = ControladorGrua(E, G, P, S, instancia_politica, instancia_politica_recuperacion, **instancia_simulacion.argumentosOperacion(len(S)))
    C.ejecutarSecuencia()
    G.registrarEvento('F')

    vector_metricas = C.instancia_metricas.recuperarInformcion()
    return {
        "nombre": instancia_politica.nombre,
        "recuperacion": instancia_politica_recuperacion.nombre,
        # Cómo se colorean los ítems: las políticas basadas en clases usan la clase de cada SKU y las demás la rotación del SKU
        "color_por": "clase" if txt_clave_politica.startswith('basada_en_clases') else "sku",
        "inicial": vector_estado_inicial,
        "eventos": G.vector_eventos,
        "tiempo_total": round(float(vector_metricas[-1]), 4),
        "reubicaciones": int(vector_metricas[6]),
        "reloj_final": round(float(G.escalar_reloj), 4),
    }


# La carrera anima como mucho estas tareas: con listas largas (datos reales de decenas de miles de tareas) los movimientos
# de la grúa pesarían cientos de MB en el navegador. La estadística sí usa la lista completa
TAREAS_ANIMACION = 2000


def generar_datos(config, vector_semillas, al_avanzar=None):
    # Graba cada política en cada réplica, usando las mismas semillas que la simulación principal
    # al_avanzar: función opcional que se llama al terminar cada réplica, sirve para informar el progreso
    instancia_simulacion = Simulacion(config)
    vector_replicas = []

    for escalar_semilla in vector_semillas:
        rnd.seed(escalar_semilla)
        SA, S = instancia_simulacion.generarSecuencias()

        vector_politicas = []
        for _, txt_clave_politica, txt_clave_recuperacion in Simulacion.escenarios(config):
            rnd.seed(Simulacion.semillaDePolitica(escalar_semilla, txt_clave_politica))
            vector_politicas.append(grabar_escenario(instancia_simulacion, txt_clave_politica, SA, S[:TAREAS_ANIMACION], txt_clave_recuperacion))

        vector_replicas.append({
            "semilla": int(escalar_semilla),
            # Demanda, clase y ranking de rotación de los SKUs de esta réplica: con un escenario generado dependen de la semilla
            **vectores_por_sku(instancia_simulacion, config),
            "secuencia": [[int(escalar_actividad), int(escalar_clase)] for escalar_actividad, escalar_clase in S[:TAREAS_ANIMACION]],
            "tareas_totales": len(S),
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
        # Demanda, clase y ranking de rotación de la primera réplica, que es la que anima la página (la plantilla lee estas claves)
        **(vectores_por_sku(instancia_simulacion, config) if not vector_replicas else
           {txt_clave: vector_replicas[0][txt_clave] for txt_clave in ("demanda_sku", "clase_sku", "rango_sku")}),
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
