# Experimentos

Cada archivo `.json` es un experimento reproducible: basta con escribir los parámetros que cambian respecto a
`asrs/config.py` (los demás toman el valor por defecto). Los nombres de las claves son los de `asrs/config.py`.

```bash
python scripts/main.py --config experimentos/ejemplo.json
python scripts/visualizar.py --config experimentos/ejemplo.json
```

Una clave que no existe se ignora y un valor inválido da un error con un mensaje en español. Guarda aquí cada
configuración que uses para un resultado de la tesis, así se puede repetir con el mismo archivo.
