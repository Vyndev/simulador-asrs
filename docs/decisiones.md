# Registro de decisiones

Una entrada por decisión que afecta a los resultados o al texto de la tesis. Es el lugar para anotar el **porqué**
mientras se recuerda: el README explica cómo funciona el simulador, no por qué se eligió así.

Plantilla para añadir una entrada nueva (las más recientes arriba):

```
## AAAA-MM-DD · Título corto
- **Contexto:** qué problema o duda había.
- **Decisión:** qué se hizo.
- **Alternativas descartadas:** y por qué.
- **Efecto en los resultados:** qué cambia si se decide distinto.
- **Dónde está en el código:** archivo o sección del README.
```

## 2026-11-02 · Minimización de varianza: fondos antes que frentes entre canales
- **Contexto:** la regla solo desempataba por profundidad entre celdas de igual criterio, y el criterio (tiempo xy) ignora la profundidad. Eso la limitaba a cada canal: el frente del canal más cercano se ocupaba antes que el fondo de uno lejano, y con ocupación superior al 50 % la capa profunda no estaba llena.
- **Decisión:** la capa libre más profunda entre todas las candidatas (de la zona, en las políticas con zonas) se elige primero; el criterio de la política desempata dentro de esa capa y después el azar.
- **Evidencia:** Lehmann y Hußmann (2023) definen la varianza mínima como la estrategia que minimiza la varianza del llenado de los canales (resumen del artículo, ficha `lehmann2023`), y la varianza máxima como la contraria; el desempate solo por canal se parece más a la máxima. Lerher et al. (2010) la describen como llenar primero la segunda profundidad, con reubicación nula hasta α = 0,5, y la tesis (cap. 7, validación) usa ese umbral como referencia. Con el desempate por canal esa referencia no se cumple. Falta contrastar con el PDF de Lehmann 2023, que no está en el equipo (la ficha es una lectura del texto, no una verificación).
- **Alternativas descartadas:** dejar el desempate solo por canal (incumple la referencia de validación); añadir un peso de profundidad al criterio (agrega un parámetro sin respaldo en la literatura).
- **Efecto en los resultados:** cambian los números de **todas** las políticas clásicas, también `aleatoria` (ahora sortea entre los canales con la capa libre más profunda) y las zonificadas (fondos antes que frentes dentro de la zona). Las corridas anteriores a esta fecha no son comparables. El agente no cambia, pero su comparación con las clásicas sí.
- **Dónde está en el código:** `PoliticaAsignacion.asignar` en `asrs/politicas/asignacion.py`; prueba `test_minimizacion_de_varianza_llena_todos_los_fondos_antes_que_los_frentes`.

## 2026-11-01 · Salidas y experimentos fuera del código
- **Contexto:** los scripts escribían CSV y HTML en la carpeta desde la que se lanzaban y el `.gitignore` los listaba uno a uno.
- **Decisión:** todo lo generado va a `salidas/` (ignorada entera). Cada experimento se guarda como JSON en `experimentos/` y se lanza con `--config`.
- **Alternativas descartadas:** seguir editando `asrs/config.py` por experimento, porque no deja rastro de qué configuración produjo cada resultado.
- **Efecto en los resultados:** ninguno en los números; solo los hace reproducibles con un archivo.
- **Dónde está en el código:** `asrs/utilidades.py` (`ruta_salida`), `asrs/config.py` (`cargar_configuracion_archivo`), `experimentos/LEEME.md`.

## Comparación pareada entre políticas
- **Contexto:** comparar políticas con secuencias distintas mezclaría el efecto de la política con el del azar.
- **Decisión:** en cada réplica todas las políticas procesan la misma secuencia de llenado y la misma de operación, con la semilla `escalar_semilla + i`. Cada política tiene una semilla derivada de su política base, no de su posición en la lista.
- **Efecto en los resultados:** las diferencias entre políticas son pareadas y las variantes replanificadas parten con los mismos desempates al azar que su variante de plan fijo.
- **Dónde está en el código:** README, sección "Modelo de réplicas"; `asrs/politicas/registro.py`.
