"""Extrae de cada fuente real solo la secuencia de demanda (SKU recuperados, en orden de tiempo, con el día del modelo) y la guarda en salidas/demanda/<fuente>.csv."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del proyecto, para importar asrs
import argparse
import json
import statistics
from collections import Counter

from asrs.config import cargar_configuracion, config_a_json, config_desde_json, dias_lista_real, recuperaciones_con_historial
from asrs.generacion.fuentes import ADAPTADORES, crear_adaptador, escribir_codigos, escribir_secuencia, extraer_secuencia
from asrs.generacion.demanda_replica import omega_de_fuente
from asrs.generacion.variabilidad import estimar_omega, recuperaciones_por_dia

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--fuente", nargs="*", default=list(ADAPTADORES), choices=list(ADAPTADORES), help="fuentes a tratar (por defecto todas)")
    analizador.add_argument("--datos", default=os.path.join(RAIZ, "..", "tesis", "datos"), help="carpeta con las fuentes (cada una con crudo/)")
    analizador.add_argument("--salida", default=os.path.join(RAIZ, "salidas", "demanda"), help="carpeta de salida")
    analizador.add_argument("--sku-con-talla", action="store_true", help="calzado: SKU = referencia-talla en vez de solo la referencia")
    analizador.add_argument("--tareas", type=int, default=None, help="escala la secuencia a exactamente N recuperaciones (0 = sin escalar; por defecto R x (Y + B) de la instancia: ciclos por hora x horas por día / 2 x (365 + 30) = 284 400, el año evaluado más 30 días de historial)")
    analizador.add_argument("--max-skus", type=int, default=None, help="reduce el surtido a N SKU con muestreo ponderado por la demanda (0 = sin reducir; por defecto 180 en las fuentes que lo necesitan)")
    analizador.add_argument("--sin-codificar", action="store_true", help="deja los nombres originales de los SKU en vez de SKU001, SKU002, ...")
    analizador.add_argument("--semilla", type=int, default=0, help="semilla del muestreo de SKU (y de la rotación, con --rotar)")
    analizador.add_argument("--rotar", action="store_true", help="rota circularmente la secuencia un desplazamiento de días sorteado con la semilla, como cada réplica (sección 8.2.3)")
    analizador.add_argument("--sin-dias", action="store_true", help="escribe solo la columna SKU, sin reescalar el calendario a Y + B días")
    argumentos = analizador.parse_args()

    os.makedirs(argumentos.salida, exist_ok=True)
    escalar_tareas = recuperaciones_con_historial(cargar_configuracion()) if argumentos.tareas is None else argumentos.tareas
    escalar_dias_modelo = None if argumentos.sin_dias else dias_lista_real(cargar_configuracion())
    for txt_fuente in argumentos.fuente:
        txt_carpeta, txt_archivo, _ = ADAPTADORES[txt_fuente]
        opciones = {"bool_sku_con_talla": True} if txt_fuente == "calzado" and argumentos.sku_con_talla else {}
        escalar_max = "adaptador" if argumentos.max_skus is None else (argumentos.max_skus or None)
        resultado = extraer_secuencia(crear_adaptador(txt_fuente, **opciones), os.path.join(argumentos.datos, txt_carpeta, "crudo", txt_archivo), escalar_max, argumentos.semilla, not argumentos.sin_codificar,
                                      escalar_tareas or None, escalar_dias_modelo, argumentos.rotar)
        txt_destino = os.path.join(argumentos.salida, f"{txt_fuente}.csv")
        escribir_secuencia(resultado.vector_skus, txt_destino, resultado.vector_dias)
        # Experimento listo para el modelo: python scripts/main.py --config salidas/demanda/<fuente>_experimento.json
        # Con las opciones por defecto el experimento usa el origen 'fuente': cada réplica trata la fuente con su semilla (muestreo y rotación) y el CSV
        # queda para inspeccionar la secuencia de esta semilla. Con opciones propias (talla, surtido, volumen) el experimento fija el CSV escrito
        with open(os.path.join(argumentos.salida, f"{txt_fuente}_experimento.json"), "w", encoding="utf-8") as archivo:
            json.dump({"texto_origen_tareas": "fuente", "texto_fuente_demanda": txt_fuente, "texto_carpeta_datos": os.path.abspath(argumentos.datos)}
                      if not (opciones or argumentos.max_skus is not None or argumentos.tareas is not None or argumentos.sin_dias or argumentos.sin_codificar)
                      else {"texto_origen_tareas": "datos", "texto_tareas_csv": open(txt_destino, encoding="utf-8").read()}, archivo)
        escribir_codigos(resultado.dic_estratos, resultado.dic_codigos, os.path.join(argumentos.salida, f"{txt_fuente}_skus.csv"), resultado.set_sin_copias)
        if resultado.vector_dias is not None:
            escalar_ventana = cargar_configuracion()["escalar_dias_ventana_popularidad"]
            escalar_omega_csv = estimar_omega(recuperaciones_por_dia(resultado.vector_skus, resultado.vector_dias)[0], escalar_ventana)
            print(f"{txt_fuente}: calendario de {resultado.escalar_dias_fuente} días reescalado a {escalar_dias_modelo}"
                  + (f", rotado {resultado.escalar_desplazamiento} días" if resultado.escalar_desplazamiento is not None else "")
                  + f"; ω de esta secuencia {escalar_omega_csv:.4f}")
            if not opciones:
                # ω̂ de la fuente que usa la simulación: el de su tratamiento de referencia (semilla 0, sin rotar), igual para todas las réplicas
                escalar_omega = omega_de_fuente(txt_fuente, os.path.abspath(argumentos.datos), escalar_tareas or recuperaciones_con_historial(cargar_configuracion()),
                                                dias_lista_real(cargar_configuracion()), escalar_ventana)
                print(f"    ω̂ de la fuente = {escalar_omega:.4f} (niveles: baja {escalar_omega / 2:.4f}, real {escalar_omega:.4f}, alta {(1 + escalar_omega) / 2:.4f})")
        print(f"{txt_fuente}: {resultado.escalar_filas_leidas} filas leídas, {resultado.escalar_descartadas} descartadas "
              f"-> {resultado.escalar_lineas} líneas -> {len(resultado.vector_skus)} recuperaciones (una por unidad), {resultado.escalar_skus_distintos} SKU distintos, "
              f"{100 * resultado.escalar_repeticiones_consecutivas:.1f} % repite el SKU anterior ({txt_destino})")
        escalar_quedan, escalar_antes = resultado.escalar_recuperaciones_muestreadas, resultado.escalar_recuperaciones_antes
        m = resultado.dic_muestreo
        if "agregados" in m:
            print(f"    surtido: {m['skus']} SKU originales, {m['agregados']} agregados por muestreo; recuperaciones tras ampliar {escalar_quedan} "
                  f"({escalar_quedan / escalar_antes:.1f} veces las de la fuente)")
        else:
            print(f"    surtido: {m['skus']} SKU originales, {m['muestreados']} elegidos; recuperaciones conservadas "
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
