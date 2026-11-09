# Prototipo: dos formas de entrenar el agente del capítulo 9

Compara, en el mismo entorno Gymnasium (`entorno.py`), dos implementaciones de PPO:

- **A. Stable-Baselines3:** `MaskablePPO` de sb3-contrib con la política MLP por defecto sobre la observación aplanada (estado y todas las celdas).
- **B. Estilo CleanRL** (`ppo_candidatas.py`): PPO de un solo archivo con una red que puntúa cada candidata con los mismos pesos (ec. 9.8) y máscara de candidatas.

Mismos hiperparámetros (los de Stable-Baselines3 por defecto), 200 000 pasos, 3 semillas de entrenamiento por opción, semillas de entrenamiento (1 a 5 000) y de evaluación (100 001 a 100 010) disjuntas. La evaluación es determinista (argmax).

## Instancia

Estantería 6 × 6 × 2 (72 celdas), 20 SKU, 85 % de ocupación, lista generada de 1 000 tareas por episodio. El llenado inicial lo hace la política aleatoria para todas, de modo que todas parten del mismo estado. Todas las políticas, también el agente, aplican la regla de minimización de varianza. Recompensa −ΔT/100 s, con una penalización de 300 s por tarea perdida que con la regla nunca se activa.

```bash
python comparar.py --opcion heuristicas --salida heuristicas.json
python comparar.py --opcion A --pasos 200000 --semilla 0 --salida res_A0.json   # y B, y semillas 1 y 2
python resumir.py <carpeta con los json>
```

## Resultados (2026-11-08)

Tiempo total medio en las 10 réplicas de evaluación y diferencia pareada con la mejor heurística (más cercana a la grúa). Ninguna política perdió tareas.

| Política | Tiempo medio (s) | vs. más cercana a la grúa |
|---|---|---|
| Más cercana a la grúa | 31 562 | — |
| Más cercana al E/S | 31 868 | +1,0 % |
| Aleatoria | 32 385 | +2,6 % |
| Basada en clases | 32 568 | +3,2 % |
| Dedicada (full turnover) | 34 552 | +9,5 % |
| A, semillas 0 / 1 / 2 | 32 432 / 32 251 / 32 432 | +2,8 / +2,2 / +2,8 % |
| B, semillas 0 / 1 / 2 | 31 575 / 32 511 / 31 635 | +0,05 / +3,0 / +0,2 % |

| | A (Stable-Baselines3) | B (estilo CleanRL) |
|---|---|---|
| Parámetros | 87 817 | 14 082 |
| Entrenamiento (200 000 pasos, 1 hilo) | 23,6 min | 30,4 a 31,6 min |
| Código propio del algoritmo | unas 30 líneas | 114 líneas |
| Red por candidata (ec. 9.8) | no (MLP sobre todo el vector) | sí |

Lectura: B iguala a la mejor heurística en dos de tres semillas y A queda a la altura de la aleatoria. Ninguna la supera, lo que es esperable en esta instancia: con la regla de varianza hay unas 6 candidatas por decisión y en 1 000 tareas (menos de un día) la popularidad casi no cambia, justo la ventaja que el agente debe aprovechar. Las corridas usaron las 8 características de candidata anteriores (3 quedaban determinadas por la capa); el entorno actual usa 5.
