"""Núcleo común del tratamiento de fuentes: contrato del adaptador, ordenamiento estable, desagregación, reducción del surtido y escritura de la secuencia."""
import csv
import io
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

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
    escalar_max_skus = None     # Máximo de SKU tras reducir el surtido con muestreo ponderado por la demanda (None = sin reducir). Calzado y ecommerce usan 180
    bool_ampliar_surtido = False  # True: si la fuente tiene menos SKU que `escalar_max_skus`, se agregan SKU por muestreo hasta completarlos (ver `ampliar_surtido`)

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
    dic_estratos: dict = field(default_factory=dict)   # SKU original -> estrato: "muestra" o "descartado"
    dic_codigos: dict = field(default_factory=dict)    # SKU original -> código final de la salida; solo los SKU que quedan (vacío si no se codificó)
    dic_muestreo: dict = field(default_factory=dict)   # conteos del muestreo: skus (originales), muestreados
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
    def escalar_racha_maxima(self):
        # Largo de la racha más larga de un mismo SKU seguido
        escalar_maxima = escalar_actual = 1 if self.vector_skus else 0
        for a, b in zip(self.vector_skus, self.vector_skus[1:]):
            escalar_actual = escalar_actual + 1 if a == b else 1
            escalar_maxima = max(escalar_maxima, escalar_actual)
        return escalar_maxima

    @property
    def escalar_skus_distintos(self):
        return len(set(self.vector_skus))


def filas_csv_en_zip(txt_ruta, txt_nombre_interno, txt_delimitador=","):
    """Itera las filas (como dict) de un CSV que está dentro de un .zip, en el orden del archivo. El zip no se extrae a disco."""
    with zipfile.ZipFile(txt_ruta) as archivo_zip:
        txt_contenido = archivo_zip.read(txt_nombre_interno).decode("utf-8-sig")
    yield from csv.DictReader(io.StringIO(txt_contenido, newline=""), delimiter=txt_delimitador)


def muestrear_surtido(vector_skus, escalar_max_skus, escalar_semilla=0):
    """Reduce el surtido a `escalar_max_skus` SKU reales con muestreo aleatorio ponderado por la demanda. Cada etiqueta que queda es un SKU original; no hay grupos.
    Entrada: la secuencia desagregada (una recuperación por unidad), en orden de tiempo.
    1. Cada SKU se elige con probabilidad proporcional a su demanda (sus recuperaciones en la secuencia), sin reemplazo y hasta completar exactamente
       `escalar_max_skus`. Se usa el método de las claves exponenciales (Efraimidis y Spirakis): a cada SKU se le asigna la clave ln(u) / demanda, con u uniforme
       en (0, 1], y se eligen los de clave mayor. No hay estrato garantizado: un SKU muy demandado casi siempre sale, pero no siempre.
       Para que sea reproducible los SKU se ordenan por primera aparición y se sortea con `numpy.random.default_rng(escalar_semilla)`.
       Si el surtido ya cabe no se muestrea.
    2. Las recuperaciones de los SKU no elegidos se descartan. Las demás conservan su orden relativo y sus repeticiones seguidas.
    Devuelve la secuencia reducida, el diccionario SKU -> estrato ("muestra" o "descartado") y un diccionario con los conteos (`skus`, `muestreados`)."""
    cuenta, primera = Counter(vector_skus), {}
    for escalar_indice, txt_sku in enumerate(vector_skus):
        primera.setdefault(txt_sku, escalar_indice)
    if len(cuenta) <= escalar_max_skus:
        return list(vector_skus), {txt_sku: "muestra" for txt_sku in cuenta}, {"skus": len(cuenta), "muestreados": len(cuenta)}
    vector_orden = sorted(cuenta, key=lambda t: primera[t])
    vector_pesos = np.array([cuenta[t] for t in vector_orden], dtype=float)
    vector_claves = np.log(1.0 - np.random.default_rng(escalar_semilla).random(len(vector_orden))) / vector_pesos
    vector_elegidos = np.argsort(-vector_claves, kind="stable")[:escalar_max_skus]
    set_muestra = {vector_orden[i] for i in vector_elegidos.tolist()}
    dic_estrato = {t: "muestra" if t in set_muestra else "descartado" for t in cuenta}
    vector_reducida = [t for t in vector_skus if dic_estrato[t] != "descartado"]
    return vector_reducida, dic_estrato, {"skus": len(cuenta), "muestreados": len(set_muestra)}


def ampliar_surtido(vector_lineas, escalar_max_skus, escalar_semilla=0):
    """Completa el surtido hasta `escalar_max_skus` SKU agregando SKU por muestreo. Es el caso contrario de `muestrear_surtido`: la fuente tiene menos SKU de los que pide el modelo.
    Entrada: las líneas válidas `(tiempo, posicion, sku, cantidad)` ya ordenadas, con `tiempo` de tipo `datetime`. Cada SKU agregado copia la serie de líneas de un SKU de origen:
    1. El SKU de origen se elige con reemplazo y con probabilidad proporcional a su demanda (unidades en las líneas), con `numpy.random.default_rng(escalar_semilla)`.
       Los SKU se ordenan por primera aparición antes de sortear.
    2. La serie del SKU agregado se desplaza circularmente un número de días al azar (uniforme entre 1 y L - 1, con L los días del periodo de la fuente), de modo que
       no repite día a día a su origen y conserva su nivel y su forma de demanda. La hora del día no cambia.
    3. El SKU agregado se llama `<origen>#<n>`, con n = 1, 2, ... por origen. Los originales no cambian.
    Si el surtido ya tiene `escalar_max_skus` o más SKU, devuelve las líneas sin cambios.
    Devuelve las líneas ordenadas por (tiempo, posición, original antes que agregado), el diccionario SKU -> "original" o "agregado" y un diccionario con los conteos
    (`skus`, `agregados`)."""
    vector_orden, dic_demanda = [], Counter()
    for _, _, txt_sku, escalar_q in vector_lineas:
        if txt_sku not in dic_demanda:
            vector_orden.append(txt_sku)
        dic_demanda[txt_sku] += escalar_q
    escalar_faltan = escalar_max_skus - len(vector_orden)
    dic_estrato = {t: "original" for t in vector_orden}
    if escalar_faltan <= 0:
        return list(vector_lineas), dic_estrato, {"skus": len(vector_orden), "agregados": 0}
    generador = np.random.default_rng(escalar_semilla)
    vector_pesos = np.array([dic_demanda[t] for t in vector_orden], dtype=float)
    vector_origenes = generador.choice(len(vector_orden), size=escalar_faltan, replace=True, p=vector_pesos / vector_pesos.sum())
    fecha_primera, fecha_ultima = vector_lineas[0][0].date(), vector_lineas[-1][0].date()
    escalar_dias = (fecha_ultima - fecha_primera).days + 1
    if escalar_dias < 2:
        raise ValueError("La fuente cubre un solo día: no se puede desplazar la serie de un SKU agregado.")
    vector_desplazamientos = generador.integers(1, escalar_dias, size=escalar_faltan)
    dic_lineas_de = {}
    for tiempo, escalar_posicion, txt_sku, escalar_q in vector_lineas:
        dic_lineas_de.setdefault(txt_sku, []).append((tiempo, escalar_posicion, escalar_q))
    vector_todas = [(tiempo, escalar_posicion, 0, txt_sku, escalar_q) for tiempo, escalar_posicion, txt_sku, escalar_q in vector_lineas]
    dic_copias = Counter()
    for escalar_j, (escalar_o, escalar_d) in enumerate(zip(vector_origenes.tolist(), vector_desplazamientos.tolist()), 1):
        txt_origen = vector_orden[escalar_o]
        dic_copias[txt_origen] += 1
        txt_nuevo = f"{txt_origen}#{dic_copias[txt_origen]}"
        dic_estrato[txt_nuevo] = "agregado"
        for tiempo, escalar_posicion, escalar_q in dic_lineas_de[txt_origen]:
            escalar_dia = ((tiempo.date() - fecha_primera).days + escalar_d) % escalar_dias
            vector_todas.append((datetime.combine(fecha_primera + timedelta(days=escalar_dia), tiempo.time()), escalar_posicion, escalar_j, txt_nuevo, escalar_q))
    vector_todas.sort(key=lambda t: (t[0], t[1], t[2]))
    return [(t, p, k, q) for t, p, _, k, q in vector_todas], dic_estrato, {"skus": len(vector_orden), "agregados": escalar_faltan}


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


def extraer_secuencia(adaptador, txt_ruta, escalar_max_skus="adaptador", escalar_semilla=0, bool_codificar=False, escalar_tareas=None):
    """Lee la fuente con el adaptador y devuelve la secuencia de SKU ordenada por tiempo.
    El orden es estable: las filas con el mismo tiempo conservan el orden que tenían en el archivo original.
    Si hay un máximo de SKU (por defecto el del adaptador; None lo desactiva) el surtido se reduce con muestreo ponderado por la demanda después de
    filtrar, ordenar y desagregar (ver `muestrear_surtido`); `escalar_semilla` es la semilla del muestreo. Si se da `escalar_tareas`, después se escala el volumen a exactamente esa cantidad de
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
    vector_mayores = [(q, txt_sku, tiempo) for tiempo, _, txt_sku, q in sorted(vector_validas, key=lambda t: (-t[3], t[0], t[1]))[:10]]
    escalar_lineas = len(vector_validas)       # líneas de la fuente, antes de agregar SKU
    escalar_antes = sum(t[3] for t in vector_validas)       # recuperaciones de la fuente, antes de reducir o agregar SKU
    if escalar_max_skus == "adaptador":
        escalar_max_skus = adaptador.escalar_max_skus
    dic_estratos, dic_muestreo, bool_ampliado = None, None, False
    if escalar_max_skus is not None and adaptador.bool_ampliar_surtido and vector_validas and len({t[2] for t in vector_validas}) < escalar_max_skus:
        vector_validas, dic_estratos, dic_muestreo = ampliar_surtido(vector_validas, escalar_max_skus, escalar_semilla)
        bool_ampliado = True
    # Desagregación completa: una línea de cantidad q son q recuperaciones seguidas del SKU, en la posición de su línea
    vector_skus = [txt_sku for _, _, txt_sku, escalar_q in vector_validas for _ in range(escalar_q)]
    if bool_ampliado:
        pass
    elif escalar_max_skus is not None:
        vector_skus, dic_estratos, dic_muestreo = muestrear_surtido(vector_skus, escalar_max_skus, escalar_semilla)
    else:
        dic_estratos = {t: "muestra" for t in set(vector_skus)}
        dic_muestreo = {"skus": len(dic_estratos), "muestreados": len(dic_estratos)}
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
