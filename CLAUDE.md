# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Proyecto de tesis: simulador de un ASRS (grúa, estantería con canales de varias capas, punto E/S) que compara políticas de asignación y recuperación sobre réplicas reproducibles. Código, comentarios, documentación y mensajes de error están en **español**; mantén esa convención (identificadores con prefijos como `escalar_`, `vector_`, `texto_`, `matriz`).

## Comandos

```bash
pip install -e .                                   # instala el paquete asrs (numpy, pandas); extra [rl] para el agente
python scripts/main.py [--config experimentos/x.json]   # simulación completa -> salidas/*.csv
python scripts/visualizar.py --servir [--puerto N --sin-abrir]   # herramienta web interactiva
python scripts/visualizar.py --abrir               # HTML autónomo en salidas/visualizacion.html
python -m unittest discover -s tests               # todas las pruebas
python -m unittest tests.test_politicas            # un módulo
python -m unittest tests.test_politicas.NombreClase.test_metodo   # una prueba
```

No hay linter configurado. Un año completo (525 600 tareas) tarda ~3,5 min por política; para probar usa listas cortas (las pruebas usan 72 tareas). Todo lo generado va a `salidas/` (ignorada por git).

## Arquitectura

Dependencias solo hacia abajo: `simulacion` → `control` → `politicas` → `dominio` / `metricas`.

- `asrs/config.py`: todos los parámetros (claves `escalar_*`, `vector_*`, `texto_*`), validación y carga desde JSON. Los experimentos reproducibles se guardan como JSON en `experimentos/` con solo las claves que cambian; se lanzan con `--config`.
- `asrs/generacion/`: el generador produce un **escenario** (`EscenarioDemanda`: surtido abierto con curva de Bass, estacionalidad, reposición EOQ, techo de ocupación, calentamiento de 30 días) y las políticas solo ven la lista de tareas `(actividad, SKU)` más la popularidad móvil. El escenario es el mismo para todas las políticas de una réplica. Con datos reales (CSV `SKU`,`Task`) no hay escenario y las variantes replanificadas se comportan como las de plan fijo.
- `asrs/politicas/`: subclases de `PoliticaAsignacion` que implementan `matrizCriterio`; se registran en `registro.py` y su clave se añade a `vector_politicas_asignacion`. Hay 7 políticas de asignación (basada en clases y full_turnover con variante de plan fijo y replanificada) y 2 de recuperación (FCFS estricto / entre accesibles). Reubicar es almacenar un ítem ya en la estantería.
- `asrs/control/controlador_grua.py` ejecuta la secuencia con las políticas; `asrs/cinematica.py` da el perfil trapezoidal de la grúa.
- `asrs/simulacion/`: réplicas (semilla `escalar_semilla + i`), ejecución en paralelo, estadística pareada (IC 95 % con t de Student) y CSV.
- `asrs/agente/`: capa de observación para el agente de aprendizaje (perfil de SKU y celdas candidatas); no cambia la simulación.
- `asrs/visualizacion/`: servidor local + `plantilla.html` (Three.js incluido en `vendor/`, sin internet).

## Reglas del modelo que no son evidentes

- **Comparación pareada:** en cada réplica todas las políticas procesan la misma secuencia de llenado y de operación; cada política tiene semilla derivada de su política base en el registro (la replanificada usa la de la de plan fijo), no de su posición en la lista. No rompas esto al tocar semillas.
- **Minimización de varianza** es fija en las políticas clásicas (entre celdas equivalentes, la libre más profunda y accesible; desempate al azar). El agente **no** está sujeto a ella: es libre de elegir cualquier celda libre y accesible.
- La cercanía entre celdas es tiempo Chebyshev en el plano xy y ignora la profundidad.
- Con la configuración por defecto, los resultados deben coincidir exactamente con corridas previas; cambios que alteren números requieren una entrada en `docs/decisiones.md`.

## Documentación

`README.md` es extenso y es la referencia del modelo (parámetros por defecto, claves de configuración, limitaciones conocidas, herramienta visual). `docs/decisiones.md` registra el **porqué** de cada decisión que afecte resultados o la tesis (plantilla al inicio, entradas recientes arriba): añade una entrada cuando tomes una. Los planes con fecha están en `docs/`.
