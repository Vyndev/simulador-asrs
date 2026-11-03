"""Núcleo común del tratamiento de fuentes: contrato del adaptador, ordenamiento estable, desagregación, reducción del surtido y escritura de la secuencia."""
import csv
import io
import zipfile
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Registro:
    # Una fila cruda ya interpretada por el adaptador. Si `txt_motivo` no es None la fila se descarta y el motivo se cuenta
    txt_sku: str = ""
    tiempo: object = None       # cualquier valor comparable (datetime, número, texto ISO); solo sirve para ordenar
    txt_motivo: object = None
    escalar_cantidad: int = 1   # unidades de la línea: la secuencia repite el SKU esta cantidad de veces seguidas


def leer_cantidad(txt_valor):
    """Interpreta la cantidad de una línea. Devuelve (cantidad entera positiva, None) o (None, motivo de descarte).
    Una cantidad no entera es un error, no se redondea: la desagregación exige unidades enteras."""
    try:
        escalar = float(txt_valor)
    except ValueError:
        return None, "cantidad ilegible"
    if escalar != escalar or escalar in (float("inf"), float("-inf")):
        return None, "cantidad ilegible"
    if escalar != int(escalar):
        raise ValueError(f"Cantidad no entera '{txt_valor}': la desagregación necesita unidades enteras; no se redondea.")
    if escalar == 0:
        return None, "cantidad igual a cero"
    if escalar < 0:
        return None, "cantidad negativa"
    return int(escalar), None


class Adaptador:
    """Contrato de una fuente. Las subclases definen `filas` e `interpretar`."""
    escalar_max_skus = None     # Máximo de SKU tras reducir el surtido con muestreo estratificado (None = sin reducir). Calzado y ecommerce usan 180

    def filas(self, txt_ruta):
        """Itera las filas del archivo crudo en su orden original, cada una como lo que `interpretar` espere (p. ej. un dict)."""
        raise NotImplementedError

    def interpretar(self, fila):
        """Devuelve un `Registro`: el SKU, el tiempo, la cantidad (entera positiva, ver `leer_cantidad`) y, si la fila no es demanda, el motivo de descarte.
        Los filtros se evalúan en el orden en que el adaptador los escribe y solo se cuenta el primer motivo que se cumpla."""
        raise NotImplementedError


@dataclass
class ResultadoExtraccion:
    vector_skus: list                                  # la secuencia desagregada (un SKU por unidad), en orden de tiempo
    escalar_filas_leidas: int = 0
    dic_descartes: dict = field(default_factory=dict)  # motivo -> número de filas
    dic_estratos: dict = field(default_factory=dict)   # SKU original -> estrato: "individual", "muestra" o "descartado"
    dic_codigos: dict = field(default_factory=dict)    # SKU original -> código final de la salida; solo los SKU que quedan (vacío si no se codificó)
    dic_muestreo: dict = field(default_factory=dict)   # conteos del muestreo: individuales, cola, muestreados
    escalar_recuperaciones_antes: int = 0              # recuperaciones de la secuencia desagregada antes de reducir el surtido
    escalar_recuperaciones_muestreadas: int = 0        # recuperaciones tras reducir el surtido y antes de escalar el volumen
    escalar_total: object = None                       # volumen al que se escaló la secuencia (None si no se escaló)
    set_sin_copias: set = field(default_factory=set)   # SKU que se quedaron sin ninguna copia al escalar a la baja
    escalar_lineas: int = 0                            # líneas con demanda tras filtrar (antes de desagregar)
    vector_lineas_mayores: list = field(default_factory=list)   # las líneas de mayor cantidad: (cantidad, SKU original, tiempo)

    @property
    def escalar_descartadas(self):
        return sum(self.dic_descartes.values())

    @property
    def escalar_repeticiones_consecutivas(self):
        # Fracción de recuperaciones que repiten el SKU de la anterior
        return sum(1 for a, b in zip(self.vector_skus, self.vector_skus[1:]) if a == b) / max(1, len(self.vector_skus) - 1)

    @property
    def escalar_skus_distintos(self):
        return len(set(self.vector_skus))


def filas_csv_en_zip(txt_ruta, txt_nombre_interno, txt_delimitador=","):
    """Itera las filas (como dict) de un CSV que está dentro de un .zip, en el orden del archivo. El zip no se extrae a disco."""
    with zipfile.ZipFile(txt_ruta) as archivo_zip:
        txt_contenido = archivo_zip.read(txt_nombre_interno).decode("utf-8-sig")
    yield from csv.DictReader(io.StringIO(txt_contenido, newline=""), delimiter=txt_delimitador)


def muestrear_surtido(vector_skus, escalar_max_skus, escalar_semilla=0, escalar_tope=0.3):
    """Reduce el surtido a `escalar_max_skus` SKU reales con muestreo estratificado. Cada etiqueta que queda es un SKU original; no hay grupos.
    Entrada: la secuencia desagregada (una recuperación por unidad), en orden de tiempo.
    1. Estrato individual (la cabeza): los SKU con demanda mayor que `escalar_tope` x n / `escalar_max_skus` (n = recuperaciones de la secuencia),
       de mayor a menor demanda (a igual demanda, el que aparece antes) y como mucho `escalar_max_skus` - 1.
    2. Estrato de muestra (la cola): del resto se eligen al azar, sin reemplazo, exactamente `escalar_max_skus` - (individuales) SKU. Para que sea
       reproducible la cola se ordena por primera aparición y se muestrea con `numpy.random.default_rng(escalar_semilla).choice(..., replace=False)`.
       Si el surtido ya cabe (la cola cabría entera en el cupo) no se muestrea.
    3. Las recuperaciones de los SKU no elegidos se descartan. Las demás conservan su orden relativo y sus repeticiones seguidas.
    Devuelve la secuencia reducida, el diccionario SKU -> estrato ("individual", "muestra" o "descartado") y un diccionario con
    los conteos (`individuales`, `cola`, `muestreados`)."""
    cuenta, primera = Counter(vector_skus), {}
    for escalar_indice, txt_sku in enumerate(vector_skus):
        primera.setdefault(txt_sku, escalar_indice)
    if len(cuenta) <= escalar_max_skus:
        return list(vector_skus), {txt_sku: "individual" for txt_sku in cuenta}, {"individuales": len(cuenta), "cola": 0, "muestreados": 0}
    escalar_maximo = escalar_tope * len(vector_skus) / escalar_max_skus
    vector_individuales = sorted((t for t, v in cuenta.items() if v > escalar_maximo), key=lambda t: (-cuenta[t], primera[t]))[:escalar_max_skus - 1]
    set_individuales = set(vector_individuales)
    vector_cola = sorted((t for t in cuenta if t not in set_individuales), key=lambda t: primera[t])
    escalar_cupo = escalar_max_skus - len(vector_individuales)
    # Aquí la cola siempre supera el cupo (si el surtido cupiera, ya se devolvió arriba), así que siempre se muestrea
    vector_indices = np.random.default_rng(escalar_semilla).choice(len(vector_cola), size=escalar_cupo, replace=False)
    vector_muestra = [vector_cola[i] for i in sorted(vector_indices.tolist())]
    set_muestra = set(vector_muestra)
    dic_estrato = {t: "individual" if t in set_individuales else "muestra" if t in set_muestra else "descartado" for t in cuenta}
    vector_reducida = [t for t in vector_skus if dic_estrato[t] != "descartado"]
    return vector_reducida, dic_estrato, {"individuales": len(vector_individuales), "cola": len(vector_cola), "muestreados": len(vector_muestra)}


def escalar_secuencia(vector_skus, escalar_total):
    """Escala la secuencia a exactamente `escalar_total` recuperaciones, sin azar. La recuperación j (de n) se copia c(j+1) - c(j) veces
    seguidas, con c(j) = (j * N + n // 2) // n en aritmética entera. El total es c(n) = N; con N < n cada recuperación aparece 0 o 1 veces
    (adelgazamiento uniforme) y con N > n, floor(N / n) o ceil(N / n) veces. Conserva el orden, las proporciones de cada SKU (hasta una
    unidad de redondeo) y las rachas proporcionalmente."""
    escalar_n = len(vector_skus)
    if escalar_n == 0:
        raise ValueError("No hay recuperaciones que escalar.")
    vector_escalada, escalar_previo = [], 0
    for escalar_j, txt_sku in enumerate(vector_skus):
        escalar_siguiente = ((escalar_j + 1) * escalar_total + escalar_n // 2) // escalar_n
        if escalar_siguiente > escalar_previo:
            vector_escalada.extend([txt_sku] * (escalar_siguiente - escalar_previo))
        escalar_previo = escalar_siguiente
    return vector_escalada


def codificar_skus(vector_skus):
    """Sustituye cada etiqueta por un código SKU001, SKU002, ... en orden de demanda decreciente (a igual demanda, el que aparece antes).
    Devuelve la secuencia codificada y el diccionario etiqueta -> código."""
    cuenta, primera = Counter(vector_skus), {}
    for escalar_indice, txt_sku in enumerate(vector_skus):
        primera.setdefault(txt_sku, escalar_indice)
    vector_orden = sorted(cuenta, key=lambda s: (-cuenta[s], primera[s]))
    escalar_ancho = max(3, len(str(len(vector_orden))))
    dic_codigo = {txt_sku: f"SKU{i + 1:0{escalar_ancho}d}" for i, txt_sku in enumerate(vector_orden)}
    return [dic_codigo[txt_sku] for txt_sku in vector_skus], dic_codigo


def extraer_secuencia(adaptador, txt_ruta, escalar_max_skus="adaptador", escalar_semilla=0, bool_codificar=False, escalar_tope=0.3, escalar_tareas=None):
    """Lee la fuente con el adaptador y devuelve la secuencia de SKU ordenada por tiempo.
    El orden es estable: las filas con el mismo tiempo conservan el orden que tenían en el archivo original.
    Si hay un máximo de SKU (por defecto el del adaptador; None lo desactiva) el surtido se reduce con muestreo estratificado después de
    filtrar, ordenar y desagregar (ver `muestrear_surtido`); `escalar_tope` (0,3 por defecto) fija el umbral de SKU individuales y
    `escalar_semilla` la semilla del muestreo. Si se da `escalar_tareas`, después se escala el volumen a exactamente esa cantidad de
    recuperaciones (ver `escalar_secuencia`) y por último se codifica."""
    vector_validas, cuenta_motivos, escalar_leidas = [], Counter(), 0
    for escalar_posicion, fila in enumerate(adaptador.filas(txt_ruta)):
        escalar_leidas += 1
        registro = adaptador.interpretar(fila)
        if registro.txt_motivo is not None:
            cuenta_motivos[registro.txt_motivo] += 1
        else:
            vector_validas.append((registro.tiempo, escalar_posicion, registro.txt_sku, registro.escalar_cantidad))
    vector_validas.sort(key=lambda t: (t[0], t[1]))
    # Desagregación completa: una línea de cantidad q son q recuperaciones seguidas del SKU, en la posición de su línea
    vector_skus = [txt_sku for _, _, txt_sku, escalar_q in vector_validas for _ in range(escalar_q)]
    vector_mayores = [(q, txt_sku, tiempo) for tiempo, _, txt_sku, q in sorted(vector_validas, key=lambda t: (-t[3], t[0], t[1]))[:10]]
    escalar_lineas = len(vector_validas)
    escalar_antes = len(vector_skus)
    if escalar_max_skus == "adaptador":
        escalar_max_skus = adaptador.escalar_max_skus
    if escalar_max_skus is not None:
        vector_skus, dic_estratos, dic_muestreo = muestrear_surtido(vector_skus, escalar_max_skus, escalar_semilla, escalar_tope)
    else:
        dic_estratos = {t: "individual" for t in set(vector_skus)}
        dic_muestreo = {"individuales": len(dic_estratos), "cola": 0, "muestreados": 0}
    escalar_muestreadas, set_sin_copias = len(vector_skus), set()
    if escalar_tareas is not None:
        vector_skus = escalar_secuencia(vector_skus, escalar_tareas)
        set_sin_copias = set(dic_estratos) - set(vector_skus) - {t for t, e in dic_estratos.items() if e == "descartado"}
    dic_codigos = {}
    if bool_codificar:
        vector_skus, dic_codigo = codificar_skus(vector_skus)
        dic_codigos = dict(dic_codigo)       # los SKU originales que quedan son sus propias etiquetas
    return ResultadoExtraccion(vector_skus, escalar_leidas, dict(cuenta_motivos), dic_estratos, dic_codigos, dic_muestreo,
                               escalar_recuperaciones_antes=escalar_antes, escalar_recuperaciones_muestreadas=escalar_muestreadas,
                               escalar_total=escalar_tareas, set_sin_copias=set_sin_copias, escalar_lineas=escalar_lineas,
                               vector_lineas_mayores=vector_mayores)


def escribir_secuencia(vector_skus, txt_ruta):
    """Escribe la secuencia como CSV de una columna con encabezado `SKU`. `leer_tareas` lo lee como una lista de recuperaciones."""
    with open(txt_ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo, lineterminator="\n")
        escritor.writerow(["SKU"])
        for txt_sku in vector_skus:
            escritor.writerow([txt_sku])


def escribir_codigos(dic_estratos, dic_codigos, txt_ruta, set_sin_copias=frozenset()):
    """Escribe la correspondencia de TODOS los SKU originales: SKU_original, SKU (el código; vacío si se descartó) y estrato.
    Si no se codificó (`dic_codigos` vacío) la columna SKU repite el nombre original de los que quedan.
    Un SKU que perdió todas sus copias al escalar el volumen (`set_sin_copias`) conserva su estrato, queda con SKU vacío y se marca
    'sin_copias' en una cuarta columna, que solo existe si hay alguno."""
    def txt_final(txt_original):
        if dic_estratos[txt_original] == "descartado" or txt_original in set_sin_copias:
            return ""
        return dic_codigos.get(txt_original, txt_original) if dic_codigos else txt_original
    with open(txt_ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo, lineterminator="\n")
        escritor.writerow(["SKU_original", "SKU", "estrato"] + (["sin_copias"] if set_sin_copias else []))
        for txt_original in sorted(dic_estratos, key=lambda t: (txt_final(t) == "", txt_final(t), t)):
            escritor.writerow([txt_original, txt_final(txt_original), dic_estratos[txt_original]]
                              + (["sin_copias" if txt_original in set_sin_copias else ""] if set_sin_copias else []))
