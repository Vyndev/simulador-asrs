"""Convierte la base de datos de Access (tablas Q y D) en una lista de tareas para la herramienta visual.

Las tablas se interpretan así:
  Q  entradas: lotes que llegan en cada periodo T con una cantidad Q de unidades (T = 0 es el inventario inicial), con su SKU en item_SKU
  D  salidas: demanda de cada periodo T, con la cantidad Q de unidades; el SKU es el primer número del código del lote (por ejemplo 10_3_1 es del SKU 10)
El simulador mueve un ítem por tarea, así que cada `unidades_por_tarea` unidades se vuelven una tarea (una carga equivalente).
Las unidades que no completan una carga se acumulan por SKU y actividad, de modo que los totales se conservan.

Uso:
  python herramientas/bd_a_tareas.py --accdb ruta/base.accdb --unidades 14 --salida datos/tareas_bd.csv
  python herramientas/bd_a_tareas.py --q Q.csv --d D.csv --unidades 14 --salida datos/tareas_bd.csv
"""
import argparse
import os
import subprocess
import sys
import tempfile

import pandas as pd


def exportar_accdb(txt_ruta_accdb, txt_carpeta_destino):
    # Exporta las tablas Q y D a CSV leyendo una copia del archivo con el controlador ODBC de Access (solo Windows)
    txt_script = f"""
$copia = Join-Path '{txt_carpeta_destino}' 'copia.accdb'
Copy-Item -LiteralPath '{txt_ruta_accdb}' -Destination $copia -Force
$c = New-Object System.Data.Odbc.OdbcConnection("Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};Dbq=$copia;ReadOnly=1")
$c.Open()
foreach ($n in 'Q','D') {{
  $da = New-Object System.Data.Odbc.OdbcDataAdapter("SELECT * FROM [$n]", $c)
  $dt = New-Object System.Data.DataTable
  [void]$da.Fill($dt)
  $dt | Export-Csv -Path (Join-Path '{txt_carpeta_destino}' "$n.csv") -NoTypeInformation -Encoding UTF8
}}
$c.Close()
"""
    resultado = subprocess.run(["powershell", "-NoProfile", "-Command", txt_script], capture_output=True, text=True)
    if resultado.returncode != 0:
        raise SystemExit("No pude leer el archivo de Access. Exporta las tablas Q y D a CSV y usa --q y --d.\n" + resultado.stderr.strip())
    return os.path.join(txt_carpeta_destino, "Q.csv"), os.path.join(txt_carpeta_destino, "D.csv")


def construir_tareas(df_entradas, df_salidas, escalar_unidades_por_tarea, booleano_inventario_inicial=True, booleano_limitar_salidas=False):
    # Devuelve la lista de tareas (actividad, SKU) y un resumen. Cada periodo: primero las entradas y luego las salidas
    # Con booleano_limitar_salidas, una salida nunca supera las existencias del SKU en ese momento (la lista queda coherente con el inventario)
    df_q = df_entradas.assign(T=df_entradas["T"].astype(int), Q=df_entradas["Q"].astype(int), sku=df_entradas["item_SKU"].astype(str).str.split("_").str[0])
    df_d = df_salidas.assign(T=df_salidas["T"].astype(int), Q=df_salidas["Q"].astype(int), sku=df_salidas["item_SKU"].astype(str).str.split("_").str[0])
    escalar_primero = 0 if booleano_inventario_inicial else 1
    acumulado, existencias, tareas = {}, {}, []
    escalar_sin_existencias = 0
    escalar_unidades_salidas = 0
    for escalar_t in range(escalar_primero, int(max(df_q["T"].max(), df_d["T"].max())) + 1):
        for txt_actividad, df in (("A", df_q), ("R", df_d)):
            for txt_sku, escalar_cantidad in df[df["T"] == escalar_t].groupby("sku", sort=False)["Q"].sum().items():
                if txt_actividad == "A":
                    existencias[txt_sku] = existencias.get(txt_sku, 0) + escalar_cantidad
                else:
                    escalar_unidades_salidas += escalar_cantidad
                    escalar_sin_existencias += max(0, escalar_cantidad - existencias.get(txt_sku, 0))
                    if booleano_limitar_salidas:
                        escalar_cantidad = min(escalar_cantidad, existencias.get(txt_sku, 0))
                    existencias[txt_sku] = max(0, existencias.get(txt_sku, 0) - escalar_cantidad)
                clave = (txt_actividad, txt_sku)
                acumulado[clave] = acumulado.get(clave, 0) + escalar_cantidad
                escalar_tareas = int(acumulado[clave] // escalar_unidades_por_tarea)
                acumulado[clave] -= escalar_tareas * escalar_unidades_por_tarea
                tareas.extend([(txt_actividad, txt_sku)] * escalar_tareas)
    resumen = {
        "tareas": len(tareas),
        "skus": len({s for _, s in tareas}),
        "pct_almacenar": round(100 * sum(1 for a, _ in tareas if a == "A") / max(1, len(tareas)), 1),
        "pct_salidas_sin_existencias": round(100 * escalar_sin_existencias / max(1, escalar_unidades_salidas), 1),
    }
    return tareas, resumen


def main():
    analizador = argparse.ArgumentParser(description="Convierte las tablas Q y D de Access en una lista de tareas CSV.")
    analizador.add_argument("--accdb", help="archivo de Access (solo Windows)")
    analizador.add_argument("--q", help="CSV de la tabla Q")
    analizador.add_argument("--d", help="CSV de la tabla D")
    analizador.add_argument("--unidades", type=float, default=14, help="unidades que equivalen a una tarea (por defecto 14)")
    analizador.add_argument("--sin-inventario-inicial", action="store_true", help="no incluir el inventario del periodo 0 como entradas")
    analizador.add_argument("--limitar-salidas", action="store_true", help="recorta las salidas a las existencias del SKU en ese momento")
    analizador.add_argument("--salida", default=os.path.join("datos", "tareas_bd.csv"))
    argumentos = analizador.parse_args()
    if argumentos.unidades <= 0:
        raise SystemExit("--unidades debe ser mayor que cero.")
    with tempfile.TemporaryDirectory() as txt_carpeta:
        if argumentos.accdb:
            txt_q, txt_d = exportar_accdb(os.path.abspath(argumentos.accdb), txt_carpeta)
        elif argumentos.q and argumentos.d:
            txt_q, txt_d = argumentos.q, argumentos.d
        else:
            raise SystemExit("Indica --accdb o bien --q y --d.")
        tareas, resumen = construir_tareas(pd.read_csv(txt_q, dtype=str), pd.read_csv(txt_d, dtype=str), argumentos.unidades, not argumentos.sin_inventario_inicial, argumentos.limitar_salidas)
    os.makedirs(os.path.dirname(os.path.abspath(argumentos.salida)), exist_ok=True)
    with open(argumentos.salida, "w", encoding="utf-8", newline="\n") as archivo:
        archivo.write("actividad;sku\n" + "".join(f"{a};{s}\n" for a, s in tareas))
    print(f"{resumen['tareas']} tareas, {resumen['skus']} SKUs, {resumen['pct_almacenar']} % almacenar -> {argumentos.salida}")
    print(f"Salidas que superan las existencias del SKU en ese momento: {resumen['pct_salidas_sin_existencias']} % de las unidades.")
    if resumen["tareas"] > 5000:
        print("Aviso: la herramienta acepta hasta 5000 tareas; sube --unidades.")


if __name__ == "__main__":
    main()
