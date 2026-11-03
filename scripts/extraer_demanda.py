"""Extrae de cada fuente real solo la secuencia de demanda (SKU recuperados, en orden de tiempo) y la guarda en salidas/demanda/<fuente>.csv."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del proyecto, para importar asrs
import argparse
import json
import statistics
from collections import Counter

from asrs.config import cargar_configuracion, config_a_json, config_desde_json, recuperaciones_con_historial
from asrs.generacion.fuentes import ADAPTADORES, crear_adaptador, escribir_codigos, escribir_secuencia, extraer_secuencia

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--fuente", nargs="*", default=list(ADAPTADORES), choices=list(ADAPTADORES), help="fuentes a tratar (por defecto todas)")
    analizador.add_argument("--datos", default=os.path.join(RAIZ, "..", "tesis", "datos"), help="carpeta con las fuentes (cada una con crudo/)")
    analizador.add_argument("--salida", default=os.path.join(RAIZ, "salidas", "demanda"), help="carpeta de salida")
    analizador.add_argument("--sku-con-talla", action="store_true", help="calzado: SKU = referencia-talla en vez de solo la referencia")
    analizador.add_argument("--tareas", type=int, default=None, help="escala la secuencia a exactamente N recuperaciones (0 = sin escalar; por defecto R x (Y + B) de la instancia: ciclos por hora x horas por día / 2 x (365 + 30) = 284 400, el año evaluado más 30 días de historial)")
    analizador.add_argument("--max-skus", type=int, default=None, help="reduce el surtido a N SKU con muestreo estratificado (0 = sin reducir; por defecto 180 en las fuentes que lo necesitan)")
    analizador.add_argument("--tope", type=float, default=0.3, help="umbral de SKU individuales, en veces la demanda media por SKU (n / máximo de SKU); por defecto 0,3")
    analizador.add_argument("--sin-codificar", action="store_true", help="deja los nombres originales de los SKU en vez de SKU001, SKU002, ...")
    analizador.add_argument("--semilla", type=int, default=0, help="semilla del muestreo de la cola")
    argumentos = analizador.parse_args()

    os.makedirs(argumentos.salida, exist_ok=True)
    escalar_tareas = recuperaciones_con_historial(cargar_configuracion()) if argumentos.tareas is None else argumentos.tareas
    for txt_fuente in argumentos.fuente:
        txt_carpeta, txt_archivo, _ = ADAPTADORES[txt_fuente]
        opciones = {"bool_sku_con_talla": True} if txt_fuente == "calzado" and argumentos.sku_con_talla else {}
        escalar_max = "adaptador" if argumentos.max_skus is None else (argumentos.max_skus or None)
        resultado = extraer_secuencia(crear_adaptador(txt_fuente, **opciones), os.path.join(argumentos.datos, txt_carpeta, "crudo", txt_archivo), escalar_max, argumentos.semilla, not argumentos.sin_codificar,
                                      argumentos.tope, escalar_tareas or None)
        txt_destino = os.path.join(argumentos.salida, f"{txt_fuente}.csv")
        escribir_secuencia(resultado.vector_skus, txt_destino)
        # Experimento listo para el modelo: python scripts/main.py --config salidas/demanda/<fuente>_experimento.json
        with open(os.path.join(argumentos.salida, f"{txt_fuente}_experimento.json"), "w", encoding="utf-8") as archivo:
            json.dump({"texto_origen_tareas": "datos", "texto_tareas_csv": open(txt_destino, encoding="utf-8").read()}, archivo)
        escribir_codigos(resultado.dic_estratos, resultado.dic_codigos, os.path.join(argumentos.salida, f"{txt_fuente}_skus.csv"), resultado.set_sin_copias)
        print(f"{txt_fuente}: {resultado.escalar_filas_leidas} filas leídas, {resultado.escalar_descartadas} descartadas "
              f"-> {resultado.escalar_lineas} líneas -> {len(resultado.vector_skus)} recuperaciones (una por unidad), {resultado.escalar_skus_distintos} SKU distintos, "
              f"{100 * resultado.escalar_repeticiones_consecutivas:.1f} % repite el SKU anterior ({txt_destino})")
        escalar_quedan, escalar_antes = resultado.escalar_recuperaciones_muestreadas, resultado.escalar_recuperaciones_antes
        m = resultado.dic_muestreo
        print(f"    surtido: {m['individuales']} individuales, {m['cola']} SKU en la cola, {m['muestreados']} muestreados; recuperaciones conservadas "
              f"{escalar_quedan} ({100 * escalar_quedan / escalar_antes:.1f} %), descartadas {escalar_antes - escalar_quedan} ({100 * (escalar_antes - escalar_quedan) / escalar_antes:.1f} %)")
        if resultado.escalar_total is not None:
            cuenta = sorted(Counter(resultado.vector_skus).values())
            print(f"    escalado: n antes {escalar_quedan}, N {resultado.escalar_total}, factor {resultado.escalar_total / escalar_quedan:.4f}; "
                  f"recuperaciones por SKU final: mín {cuenta[0]}, mediana {statistics.median(cuenta):g}, máx {cuenta[-1]}; "
                  f"racha máxima de un mismo SKU: {resultado.escalar_racha_maxima}; SKU sin copias: {len(resultado.set_sin_copias)}" + (f" ({', '.join(sorted(resultado.set_sin_copias))})" if resultado.set_sin_copias else ""))
        # Comprueba que el experimento carga en config y cuántas tareas tiene la lista de operación (con la reposición uno a uno)
        datos_config = config_a_json(cargar_configuracion())
        datos_config.update(texto_origen_tareas="datos", texto_tareas_csv=open(txt_destino, encoding="utf-8").read())
        try:
            config = config_desde_json(datos_config)
            set_evaluados = {sku for act, sku in config["secuencia_real"] if act == 1}
            escalar_sin_demanda = sum(1 for i, d in enumerate(config["vector_demanda_sku"]) if d > 0 and (i + 1) not in set_evaluados)
            print(f"    historial: {sum(1 for d in config['vector_demanda_sku'] if d > 0)} SKU con demanda, {escalar_sin_demanda} sin ninguna recuperación en lo evaluado; "
                  f"{sum(1 for i, d in enumerate(config['vector_demanda_sku']) if d == 0)} SKU sin demanda en el historial (nacen en lo evaluado)")
            print(f"    config: carga OK; historial de {config['escalar_tareas_historial']} recuperaciones (no se evalúan) y {len(resultado.vector_skus) - config['escalar_tareas_historial']} evaluadas; lista de operación de {len(config['secuencia_real'])} tareas (con la reposición por lotes EOQ), inventario inicial {sum(config['vector_inventario_inicial'] or [])}, {config['escalar_numero_skus']} SKU")
        except ValueError as error:
            print(f"    config: NO carga: {error}")
        print("    líneas más grandes: " + "; ".join(f"{q} u. {txt_sku} {tiempo}" for q, txt_sku, tiempo in resultado.vector_lineas_mayores))
        for txt_motivo, escalar in sorted(resultado.dic_descartes.items(), key=lambda p: -p[1]):
            print(f"    descartadas por {txt_motivo}: {escalar}")
