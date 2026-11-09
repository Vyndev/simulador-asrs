"""Resume las corridas de comparar.py: tiempo medio de evaluación, tareas perdidas, costo de entrenamiento y diferencia pareada con la mejor heurística."""
import glob
import json
import os
import sys

import numpy as np

carpeta = sys.argv[1]
heur = json.load(open(os.path.join(carpeta, "heuristicas.json")))
print("Heurísticas (10 réplicas de evaluación):")
medias = {}
for h, r in heur.items():
    medias[h] = np.mean(r["tiempo"])
    print(f"  {h:18s} tiempo medio {medias[h]:9.0f} s   perdidas {sum(r['perdidas'])}")
mejor = min(medias, key=medias.get)
t_mejor = np.array(heur[mejor]["tiempo"])
print(f"  mejor heurística: {mejor}\n")
for opcion in ("A", "B"):
    print(f"Opción {opcion}:")
    for f in sorted(glob.glob(os.path.join(carpeta, f"res_{opcion}*.json"))):
        r = json.load(open(f))
        t = np.array(r["evaluacion"]["tiempo"]); p = np.array(r["evaluacion"]["perdidas"])
        dif = (t - t_mejor) / t_mejor * 100
        curva = r["curva"]
        print(f"  {os.path.basename(f):12s} tiempo medio {t.mean():9.0f} s   perdidas {p.sum():4d} (réplicas con pérdidas: {(p > 0).sum()}/10)   "
              f"vs {mejor}: {dif.mean():+6.2f} % (mín {dif.min():+.2f}, máx {dif.max():+.2f})   "
              f"entrenamiento {r['segundos_entrenamiento'] / 60:5.1f} min, {r['parametros']} parámetros, "
              f"últimos episodios: perdidas {curva[-1]['perdidas_medias_ultimos']:.1f}")
