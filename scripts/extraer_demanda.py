"""Extrae de cada fuente real solo la secuencia de demanda (SKU recuperados, en orden de tiempo) y la guarda en salidas/demanda/<fuente>.csv."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del proyecto, para importar asrs
import argparse
import json

from asrs.generacion.fuentes import ADAPTADORES, crear_adaptador, escribir_codigos, escribir_secuencia, extraer_secuencia

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--fuente", nargs="*", default=list(ADAPTADORES), choices=list(ADAPTADORES), help="fuentes a tratar (por defecto todas)")
    analizador.add_argument("--datos", default=os.path.join(RAIZ, "..", "tesis", "datos"), help="carpeta con las fuentes (cada una con crudo/)")
    analizador.add_argument("--salida", default=os.path.join(RAIZ, "salidas", "demanda"), help="carpeta de salida")
    analizador.add_argument("--sku-con-talla", action="store_true", help="calzado: SKU = referencia-talla en vez de solo la referencia")
    analizador.add_argument("--grupos", type=int, default=None, help="agrupa los SKU en N grupos con k-means (0 = sin agrupar; por defecto el de cada fuente)")
    analizador.add_argument("--tope", type=float, default=None, help="umbral de SKU individuales al agrupar, en veces la demanda media por etiqueta (por defecto el de cada fuente: 0,1 en calzado y ecommerce, 1,5 en almacén inteligente)")
    analizador.add_argument("--sin-codificar", action="store_true", help="deja los nombres originales de los SKU en vez de SKU001, SKU002, ...")
    analizador.add_argument("--semilla", type=int, default=0, help="semilla del k-means")
    argumentos = analizador.parse_args()

    os.makedirs(argumentos.salida, exist_ok=True)
    for txt_fuente in argumentos.fuente:
        txt_carpeta, txt_archivo, _ = ADAPTADORES[txt_fuente]
        opciones = {"bool_sku_con_talla": True} if txt_fuente == "calzado" and argumentos.sku_con_talla else {}
        escalar_grupos = "adaptador" if argumentos.grupos is None else (argumentos.grupos or None)
        resultado = extraer_secuencia(crear_adaptador(txt_fuente, **opciones), os.path.join(argumentos.datos, txt_carpeta, "crudo", txt_archivo), escalar_grupos, argumentos.semilla, not argumentos.sin_codificar,
                                      "adaptador" if argumentos.tope is None else argumentos.tope)
        txt_destino = os.path.join(argumentos.salida, f"{txt_fuente}.csv")
        escribir_secuencia(resultado.vector_skus, txt_destino)
        # Experimento listo para el modelo: python scripts/main.py --config salidas/demanda/<fuente>_experimento.json
        with open(os.path.join(argumentos.salida, f"{txt_fuente}_experimento.json"), "w", encoding="utf-8") as archivo:
            json.dump({"texto_origen_tareas": "datos", "texto_tareas_csv": open(txt_destino, encoding="utf-8").read()}, archivo)
        if resultado.dic_codigos:
            escribir_codigos(resultado.dic_codigos, os.path.join(argumentos.salida, f"{txt_fuente}_skus.csv"))
        print(f"{txt_fuente}: {resultado.escalar_filas_leidas} filas leídas, {resultado.escalar_descartadas} descartadas "
              f"-> {len(resultado.vector_skus)} recuperaciones, {resultado.escalar_skus_distintos} SKU distintos ({txt_destino})")
        for txt_motivo, escalar in sorted(resultado.dic_descartes.items(), key=lambda p: -p[1]):
            print(f"    descartadas por {txt_motivo}: {escalar}")
