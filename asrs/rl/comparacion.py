"""Comparación de políticas y agentes desde los mismos estados iniciales, con diferencias pareadas contra la mejor política clásica."""
import csv

import numpy as np

from asrs.rl.entrenamiento import CLASICAS


def filas_comparacion(dic_resultados):
    # Una fila por política: tiempo medio, diferencia pareada contra la mejor clásica con su intervalo de confianza del 95 %, victorias y tareas omitidas
    # Pareado significa que se compara estado por estado: todas las políticas parten del mismo estado y ejecutan la misma lista
    txt_mejor = min((c for c in CLASICAS if c in dic_resultados), key=lambda c: dic_resultados[c]["tiempos"].mean())
    vector_mejor = dic_resultados[txt_mejor]["tiempos"]
    vector_filas = []
    for txt_clave, dic in dic_resultados.items():
        vector = dic["tiempos"]
        vector_dif = vector - vector_mejor
        escalar_error = vector_dif.std(ddof=1) / np.sqrt(len(vector_dif)) if len(vector_dif) > 1 else 0.0
        vector_filas.append({"politica": txt_clave, "tiempo": float(vector.mean()), "dif_pct": 100 * float(vector.mean() / vector_mejor.mean() - 1),
                             "ic95_pct": 100 * 1.96 * float(escalar_error) / float(vector_mejor.mean()), "ganadas": int((vector_dif < 0).sum()), "estados": len(vector),
                             "omitidas": float(dic["omitidas"].mean()) if "omitidas" in dic else float("nan"), "es_mejor": txt_clave == txt_mejor})
    return txt_mejor, vector_filas


def tabla(dic_resultados):
    txt_mejor, vector_filas = filas_comparacion(dic_resultados)
    vector_lineas = [f"{'Política':<20}{'Tiempo total (s)':>18}{'vs mejor clásica':>18}{'± IC 95 %':>11}{'Gana a la mejor':>17}{'Omitidas':>10}"]
    for f in sorted(vector_filas, key=lambda f: f["tiempo"]):
        txt_ic = "" if f["es_mejor"] else f"{f['ic95_pct']:.1f} pp"
        vector_lineas.append(f"{f['politica']:<20}{f['tiempo']:>18.0f}{f['dif_pct']:>17.1f}%{txt_ic:>11}{str(f['ganadas']) + '/' + str(f['estados']):>17}{f['omitidas']:>10.1f}")
    vector_lineas.append(f"\nMejor política clásica: {txt_mejor}. Diferencias pareadas por estado; si el intervalo no incluye 0 la diferencia es real. "
                         "Omitidas: tareas por episodio que no se pudieron ejecutar (menos es mejor, y no comparable si difiere mucho).")
    return "\n".join(vector_lineas)


def guardar_csv(dic_resultados, txt_ruta):
    _, vector_filas = filas_comparacion(dic_resultados)
    with open(txt_ruta, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(vector_filas[0].keys()), delimiter=";")
        escritor.writeheader()
        escritor.writerows(vector_filas)
