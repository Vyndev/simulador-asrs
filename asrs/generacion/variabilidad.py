"""Escenarios de variabilidad de la demanda con la distribución de Dirichlet (capítulo 8, sección 8.2.4).

La secuencia tratada de una fuente es una lista de recuperaciones (SKU, día del modelo). Cada día d tiene n_d recuperaciones y la tendencia
ρ̃_i(d) es la participación del SKU i en la ventana de W días centrada en d. La mezcla del día se sortea como p(d) ~ Dirichlet(η ρ̃(d)) y las
recuperaciones del día son un sorteo multinomial de tamaño n_d con probabilidades p(d), en orden aleatorio dentro del día. La variabilidad se
expresa con ω = 1 / (1 + η), la correlación entre dos recuperaciones de un mismo día (ec. 8.6), y ω̂ se estima por el método de los momentos (ec. 8.8).
"""
import numpy as np

NIVELES_VARIABILIDAD = ("ninguna", "baja", "real", "alta") # 'ninguna' usa la secuencia tratada tal cual; los otros tres son los niveles del factor


def recuperaciones_por_dia(vector_skus, vector_dias, escalar_dias=None):
    """Matriz (días, SKU) con las recuperaciones de cada SKU en cada día. Los SKU se numeran por orden de primera aparición.
    Devuelve la matriz y la lista de etiquetas en el orden de sus columnas."""
    dic_columna = {}
    for txt_sku in vector_skus:
        dic_columna.setdefault(txt_sku, len(dic_columna))
    vector_dias = np.asarray(vector_dias, dtype=np.int64)
    if escalar_dias is None:
        escalar_dias = int(vector_dias.max()) + 1 if len(vector_dias) else 0
    matriz = np.zeros((escalar_dias, len(dic_columna)), dtype=np.int64)
    np.add.at(matriz, (vector_dias, np.fromiter((dic_columna[t] for t in vector_skus), dtype=np.int64, count=len(vector_skus))), 1)
    return matriz, list(dic_columna)


def tendencia(matriz_dias, escalar_ventana):
    """Participación de cada SKU en la ventana de `escalar_ventana` días centrada en cada día: los días d - W//2 a d - W//2 + W - 1. Cerca del
    principio y del final de la secuencia la ventana se desplaza hacia adentro para conservar sus W días. Una fila sin recuperaciones en toda su
    ventana queda en cero."""
    escalar_dias = matriz_dias.shape[0]
    matriz_acumulada = np.vstack([np.zeros((1, matriz_dias.shape[1]), dtype=np.int64), np.cumsum(matriz_dias, axis=0)])
    vector_d = np.arange(escalar_dias)
    vector_desde = np.clip(vector_d - escalar_ventana // 2, 0, max(escalar_dias - escalar_ventana, 0))
    vector_hasta = np.minimum(vector_desde + escalar_ventana, escalar_dias)
    matriz_ventana = (matriz_acumulada[vector_hasta] - matriz_acumulada[vector_desde]).astype(float)
    vector_total = matriz_ventana.sum(axis=1, keepdims=True)
    return np.divide(matriz_ventana, vector_total, out=np.zeros_like(matriz_ventana), where=vector_total > 0)


def estimar_omega(matriz_dias, escalar_ventana):
    """ω̂ por el método de los momentos (ec. 8.8): (D_obs - D_bin) / (D_máx - D_bin), con los promedios sobre SKU y días de
    (x_i(d) - n_d ρ̃_i(d))², n_d ρ̃_i(d)(1 - ρ̃_i(d)) y n_d² ρ̃_i(d)(1 - ρ̃_i(d)). Se acota a [0, 1]."""
    matriz_rho = tendencia(matriz_dias, escalar_ventana)
    vector_n = matriz_dias.sum(axis=1, keepdims=True).astype(float)
    matriz_bernoulli = matriz_rho * (1.0 - matriz_rho)
    escalar_obs = float(np.mean((matriz_dias - vector_n * matriz_rho) ** 2))
    escalar_bin = float(np.mean(vector_n * matriz_bernoulli))
    escalar_max = float(np.mean(vector_n ** 2 * matriz_bernoulli))
    if escalar_max <= escalar_bin:
        return 0.0 # Todos los días tienen a lo sumo una recuperación: no hay pares que correlacionar
    return float(np.clip((escalar_obs - escalar_bin) / (escalar_max - escalar_bin), 0.0, 1.0))


def omega_del_nivel(txt_nivel, escalar_omega_fuente):
    """ω de cada nivel del factor: baja ω̂/2, real ω̂ y alta (1 + ω̂)/2. 'ninguna' no tiene ω (None)."""
    if txt_nivel not in NIVELES_VARIABILIDAD:
        raise ValueError(f"Variabilidad: debe ser {', '.join(repr(n) for n in NIVELES_VARIABILIDAD)}.")
    return {"ninguna": None, "baja": escalar_omega_fuente / 2, "real": escalar_omega_fuente, "alta": (1 + escalar_omega_fuente) / 2}[txt_nivel]


def mezclar_dirichlet(vector_skus, vector_dias, escalar_omega, escalar_ventana, generador):
    """Sortea una secuencia nueva con la misma tendencia y el mismo volumen diario (ec. 8.5). Con ω = 0 la mezcla del día es la tendencia y
    con ω = 1 todas las recuperaciones del día van a un solo SKU, elegido con la tendencia como probabilidad.
    Devuelve (skus, días), ordenada por día y en orden aleatorio dentro de cada día."""
    if not 0.0 <= escalar_omega <= 1.0:
        raise ValueError(f"Variabilidad: ω = {escalar_omega} debe estar entre 0 y 1.")
    matriz_dias, vector_etiquetas = recuperaciones_por_dia(vector_skus, vector_dias)
    matriz_rho = tendencia(matriz_dias, escalar_ventana)
    vector_n = matriz_dias.sum(axis=1)
    escalar_eta = np.inf if escalar_omega == 0 else (1.0 - escalar_omega) / escalar_omega
    vector_salida_skus, vector_salida_dias = [], []
    for escalar_d, escalar_nd in enumerate(vector_n.tolist()):
        if escalar_nd == 0:
            continue
        vector_rho = matriz_rho[escalar_d]
        vector_positivos = np.flatnonzero(vector_rho > 0)
        vector_base = vector_rho[vector_positivos] / vector_rho[vector_positivos].sum()
        if escalar_eta == np.inf:
            vector_p = vector_base
        elif escalar_eta == 0:
            vector_p = np.zeros(len(vector_base))
            vector_p[generador.choice(len(vector_base), p=vector_base)] = 1.0
        else:
            vector_p = generador.dirichlet(escalar_eta * vector_base)
            if not np.all(np.isfinite(vector_p)) or vector_p.sum() <= 0:
                # Con η muy pequeño la Dirichlet de numpy puede quedar sin masa por desbordamiento: se toma el límite, un solo SKU
                vector_p = np.zeros(len(vector_base))
                vector_p[generador.choice(len(vector_base), p=vector_base)] = 1.0
        vector_conteo = generador.multinomial(escalar_nd, vector_p / vector_p.sum())
        vector_dia = np.repeat(vector_positivos, vector_conteo)
        generador.shuffle(vector_dia)
        vector_salida_skus.extend(vector_etiquetas[i] for i in vector_dia.tolist())
        vector_salida_dias.extend([escalar_d] * escalar_nd)
    return vector_salida_skus, vector_salida_dias
