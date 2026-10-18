"""Genera visualizacion.html: las políticas ejecutan la misma lista de tareas en paralelo y se ve cuál termina primero."""
import argparse
import os
import webbrowser

from asrs.config import cargar_configuracion
from asrs.visualizacion.generador import escribir_html, generar_datos


if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--semilla", type=int, default=None, help="semilla de la primera réplica (por defecto la de la configuración)")
    analizador.add_argument("--replicas", type=int, default=3, help="cantidad de réplicas a incluir (por defecto 3)")
    analizador.add_argument("--salida", default="visualizacion.html", help="archivo HTML de salida")
    analizador.add_argument("--abrir", action="store_true", help="abrir el resultado en el navegador")
    argumentos = analizador.parse_args()

    config = cargar_configuracion()
    escalar_semilla = argumentos.semilla if argumentos.semilla is not None else config["escalar_semilla"]
    vector_semillas = [escalar_semilla + escalar_replica for escalar_replica in range(argumentos.replicas)]

    diccionario_datos = generar_datos(config, vector_semillas)
    escribir_html(diccionario_datos, argumentos.salida)

    txt_ruta = os.path.abspath(argumentos.salida)
    print(f"Visualización generada: {txt_ruta}")
    if argumentos.abrir:
        webbrowser.open("file:///" + txt_ruta.replace("\\", "/"))
