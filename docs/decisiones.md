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
