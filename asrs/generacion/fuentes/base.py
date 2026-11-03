"""Núcleo común del tratamiento de fuentes: contrato del adaptador, ordenamiento estable y escritura de la secuencia."""
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


class Adaptador:
    """Contrato de una fuente. Las subclases definen `filas` e `interpretar`."""
    escalar_grupos = None       # Máximo de etiquetas de SKU tras condensar el surtido con k-means (None = sin agrupar). Todas las fuentes usan 180

    def filas(self, txt_ruta):
        """Itera las filas del archivo crudo en su orden original, cada una como lo que `interpretar` espere (p. ej. un dict)."""
        raise NotImplementedError

    def interpretar(self, fila):
        """Devuelve un `Registro`: el SKU, el tiempo y, si la fila no es una recuperación válida, el motivo de descarte.
        Los filtros se evalúan en el orden en que el adaptador los escribe y solo se cuenta el primer motivo que se cumpla."""
        raise NotImplementedError


@dataclass
class ResultadoExtraccion:
    vector_skus: list                                  # la secuencia, en orden de tiempo
    escalar_filas_leidas: int = 0
    dic_descartes: dict = field(default_factory=dict)  # motivo -> número de filas
    dic_grupos: dict = field(default_factory=dict)     # SKU original -> grupo (vacío si no se agrupó)
    dic_codigos: dict = field(default_factory=dict)    # SKU original -> código final de la salida (vacío si no se codificó)

    @property
    def escalar_descartadas(self):
        return sum(self.dic_descartes.values())

    @property
    def escalar_skus_distintos(self):
        return len(set(self.vector_skus))


def filas_csv_en_zip(txt_ruta, txt_nombre_interno, txt_delimitador=","):
    """Itera las filas (como dict) de un CSV que está dentro de un .zip, en el orden del archivo. El zip no se extrae a disco."""
    with zipfile.ZipFile(txt_ruta) as archivo_zip:
        txt_contenido = archivo_zip.read(txt_nombre_interno).decode("utf-8-sig")
    yield from csv.DictReader(io.StringIO(txt_contenido, newline=""), delimiter=txt_delimitador)


def _kmeans(matriz, escalar_k, escalar_semilla, escalar_iteraciones=100):
    # K-means de Lloyd con inicialización k-means++, solo numpy. Devuelve la etiqueta de grupo de cada fila
    generador = np.random.default_rng(escalar_semilla)
    escalar_n = len(matriz)
    matriz_centros = [matriz[generador.integers(escalar_n)]]
    vector_dist = ((matriz - matriz_centros[0]) ** 2).sum(axis=1)
    for _ in range(1, escalar_k):
        escalar_total = vector_dist.sum()
        escalar_i = generador.choice(escalar_n, p=vector_dist / escalar_total) if escalar_total > 0 else generador.integers(escalar_n)
        matriz_centros.append(matriz[escalar_i])
        vector_dist = np.minimum(vector_dist, ((matriz - matriz[escalar_i]) ** 2).sum(axis=1))
    matriz_centros = np.array(matriz_centros)
    vector_etiquetas = np.full(escalar_n, -1)
    for _ in range(escalar_iteraciones):
        matriz_d = ((matriz[:, None, :] - matriz_centros[None, :, :]) ** 2).sum(axis=2)
        vector_nuevas = matriz_d.argmin(axis=1)
        if np.array_equal(vector_nuevas, vector_etiquetas):
            break
        vector_etiquetas = vector_nuevas
        for escalar_c in range(escalar_k):
            vector_miembros = vector_etiquetas == escalar_c
            if vector_miembros.any():
                matriz_centros[escalar_c] = matriz[vector_miembros].mean(axis=0)
    return vector_etiquetas


def agrupar_surtido(vector_skus, escalar_grupos, escalar_semilla=0, escalar_tope=0.1):
    """Agrupa los SKU en a lo sumo `escalar_grupos` etiquetas sin eliminar ninguna recuperación y sin crear una "masa gris".
    1. Los SKU que por sí solos superan el tope de demanda (`escalar_tope` veces la demanda media por etiqueta, n / grupos) quedan individuales;
       con el tope por defecto (0,1) la cabeza de la demanda queda individual y solo se agrupa la cola; con uno alto quedan pocos individuales
       y conservan su nombre: agruparlos con otros borraría justo la concentración que se quiere conservar.
    2. El resto (la cola) se agrupa con k-means sobre log(1 + frecuencia) y la primera, la última y la posición media de aparición
       (fracción del largo, estandarizadas): un grupo reúne SKU de popularidad y ciclo de vida parecidos.
    3. Se parte de menos grupos de los disponibles y, mientras haya cupo, se divide en dos con k-means el grupo que pasa del tope de demanda
       y, si ninguno pasa, el más disperso en el tiempo (demanda x dispersión de la posición media de sus SKU, si esta supera 0,15):
       así no quedan grupos que mezclen SKU de principio y de fin de la secuencia. Si queda cupo, se divide el de más demanda.
    Los grupos se llaman G001, G002, ... en orden de demanda total decreciente.
    Devuelve la secuencia de etiquetas y el diccionario SKU -> etiqueta."""
    dic_posiciones = {}
    for escalar_indice, txt_sku in enumerate(vector_skus):
        dic_posiciones.setdefault(txt_sku, []).append(escalar_indice)
    if len(dic_posiciones) <= escalar_grupos:
        return list(vector_skus), {txt_sku: txt_sku for txt_sku in dic_posiciones}
    escalar_largo = len(vector_skus)
    escalar_maximo = escalar_tope * escalar_largo / escalar_grupos
    # Individuales: los que pasan del tope, de mayor a menor demanda y a lo sumo grupos - 1 (siempre queda una etiqueta para la cola).
    # Con un tope muy bajo y un surtido casi uniforme pasarían todos; los de menor demanda entonces se agrupan
    vector_individuales = sorted((t for t, p in dic_posiciones.items() if len(p) > escalar_maximo), key=lambda t: (-len(dic_posiciones[t]), dic_posiciones[t][0]))[:escalar_grupos - 1]
    set_individuales = set(vector_individuales)
    vector_cola = [txt_sku for txt_sku in dic_posiciones if txt_sku not in set_individuales]
    escalar_cupo = escalar_grupos - len(vector_individuales)
    if len(vector_cola) <= escalar_cupo:      # la cola cabe sin agrupar
        return list(vector_skus), {txt_sku: txt_sku for txt_sku in dic_posiciones}

    matriz = np.array([[np.log1p(len(dic_posiciones[t])), dic_posiciones[t][0] / escalar_largo, dic_posiciones[t][-1] / escalar_largo,
                        np.mean(dic_posiciones[t]) / escalar_largo] for t in vector_cola])
    matriz = (matriz - matriz.mean(axis=0)) / np.where(matriz.std(axis=0) > 0, matriz.std(axis=0), 1)
    vector_demanda = np.array([len(dic_posiciones[t]) for t in vector_cola])
    vector_grupo = _kmeans(matriz, max(1, int(np.ceil(0.6 * escalar_cupo))), escalar_semilla)
    # Divide grupos mientras quede cupo y se pueda partir: primero los que pasan del tope y después los más dispersos en el tiempo
    vector_media = np.array([np.mean(dic_posiciones[t]) / escalar_largo for t in vector_cola])
    set_indivisibles, escalar_siguiente = set(), int(vector_grupo.max()) + 1
    while len(set(vector_grupo.tolist())) < escalar_cupo:
        dic_puntaje, dic_excede, dic_resto = {}, {}, {}
        for escalar_g in set(vector_grupo.tolist()) - set_indivisibles:
            vector_m = vector_grupo == escalar_g
            if vector_m.sum() < 2:
                continue
            escalar_d = int(vector_demanda[vector_m].sum())
            escalar_disp = float(np.sqrt(np.cov(vector_media[vector_m], aweights=vector_demanda[vector_m]))) if vector_m.sum() > 1 else 0.0
            if escalar_d > escalar_maximo:
                dic_excede[escalar_g] = escalar_d
            elif escalar_disp > 0.15:
                dic_puntaje[escalar_g] = escalar_d * escalar_disp
            else:
                dic_resto[escalar_g] = escalar_d
        dic_candidatos = dic_excede or dic_puntaje or dic_resto
        if not dic_candidatos:
            break
        escalar_g = max(dic_candidatos, key=dic_candidatos.get)
        vector_idx = np.where(vector_grupo == escalar_g)[0]
        vector_mitad = _kmeans(matriz[vector_idx], 2, escalar_semilla)
        if len(set(vector_mitad.tolist())) < 2:
            set_indivisibles.add(escalar_g)
            continue
        vector_grupo[vector_idx[vector_mitad == 1]] = escalar_siguiente
        escalar_siguiente += 1

    # Nombre por demanda total decreciente (a igual demanda, el que aparece antes)
    dic_demanda, dic_primera = {}, {}
    for txt_sku, escalar_g, escalar_d in zip(vector_cola, vector_grupo, vector_demanda):
        dic_demanda[escalar_g] = dic_demanda.get(escalar_g, 0) + int(escalar_d)
        dic_primera[escalar_g] = min(dic_primera.get(escalar_g, escalar_largo), dic_posiciones[txt_sku][0])
    vector_orden = sorted(dic_demanda, key=lambda g: (-dic_demanda[g], dic_primera[g]))
    dic_nombre = {g: f"G{i + 1:03d}" for i, g in enumerate(vector_orden)}
    dic_etiqueta = {txt_sku: txt_sku for txt_sku in vector_individuales}
    dic_etiqueta.update({txt_sku: dic_nombre[g] for txt_sku, g in zip(vector_cola, vector_grupo)})
    return [dic_etiqueta[txt_sku] for txt_sku in vector_skus], dic_etiqueta


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


def extraer_secuencia(adaptador, txt_ruta, escalar_grupos="adaptador", escalar_semilla=0, bool_codificar=False, escalar_tope=0.1):
    """Lee la fuente con el adaptador y devuelve la secuencia de SKU ordenada por tiempo.
    El orden es estable: las filas con el mismo tiempo conservan el orden que tenían en el archivo original.
    `escalar_tope` (0,1 por defecto) fija el umbral de SKU individuales de la agrupación (ver `agrupar_surtido`).
    Si hay un número de grupos (por defecto el del adaptador; None lo desactiva) se agrupa el surtido con k-means después de filtrar y ordenar."""
    vector_validas, cuenta_motivos, escalar_leidas = [], Counter(), 0
    for escalar_posicion, fila in enumerate(adaptador.filas(txt_ruta)):
        escalar_leidas += 1
        registro = adaptador.interpretar(fila)
        if registro.txt_motivo is not None:
            cuenta_motivos[registro.txt_motivo] += 1
        else:
            vector_validas.append((registro.tiempo, escalar_posicion, registro.txt_sku))
    vector_validas.sort(key=lambda t: (t[0], t[1]))
    vector_skus = [txt_sku for _, _, txt_sku in vector_validas]
    if escalar_grupos == "adaptador":
        escalar_grupos = adaptador.escalar_grupos
    dic_grupos = {}
    if escalar_grupos is not None:
        vector_skus, dic_grupos = agrupar_surtido(vector_skus, escalar_grupos, escalar_semilla, escalar_tope)
    dic_codigos = {}
    if bool_codificar:
        vector_originales = vector_skus
        vector_skus, dic_codigo = codificar_skus(vector_originales)
        # Cada SKU original apunta a su código final, pasando por su grupo si lo hubo
        dic_codigos = {txt_original: dic_codigo[txt_etiqueta] for txt_original, txt_etiqueta in (dic_grupos or {t: t for t in dic_codigo}).items()}
    return ResultadoExtraccion(vector_skus, escalar_leidas, dict(cuenta_motivos), dic_grupos, dic_codigos)


def escribir_secuencia(vector_skus, txt_ruta):
    """Escribe la secuencia como CSV de una columna con encabezado `SKU`. `leer_tareas` lo lee como una lista de recuperaciones."""
    with open(txt_ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo, lineterminator="\n")
        escritor.writerow(["SKU"])
        for txt_sku in vector_skus:
            escritor.writerow([txt_sku])


def escribir_codigos(dic_codigos, txt_ruta):
    """Escribe la correspondencia SKU original -> código de la salida, para poder rastrear qué es cada código."""
    with open(txt_ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo, lineterminator="\n")
        escritor.writerow(["SKU_original", "SKU"])
        for txt_sku, txt_codigo in sorted(dic_codigos.items(), key=lambda p: (p[1], p[0])):
            escritor.writerow([txt_sku, txt_codigo])
