"""Surtido abierto de productos con ciclo de vida de Bass y estacionalidad: intensidad y participación de la demanda de cada producto activo."""
import math

import numpy as np

from asrs.generacion.probabilidades import demanda_hausman

DIAS_POR_ANIO = 365
DIAS_POR_MES = 365 / 12 # Un mes son 365/12 días, unos 30,42
# Meses de cada grupo de pico estacional, numerados de 0 (enero) a 11: diciembre a febrero, junio a agosto y el resto
MESES_POR_GRUPO_PICO = ((11, 0, 1), (5, 6, 7), (2, 3, 4, 8, 9, 10))
DIA_SIN_CIERRE = 10 ** 9 # Día de cierre de un producto sin ciclo de vida (modelo estático)


def densidad_bass(vector_edad, escalar_duracion, escalar_razon_qp):
    # Densidad del ciclo de vida de Bass g(θ) = β (1 + r) e^(-βθ) / (1 + r e^(-βθ))², con r = q / p y β = 2 ln r / L
    # La curva es simétrica alrededor de L / 2: el pico está ahí y en los extremos vale 4 r / (1 + r)² del pico
    # Vale 0 fuera de [0, L). Con duración infinita no hay ciclo y la densidad es constante e igual a 1
    vector_edad = np.asarray(vector_edad, dtype=float)
    if np.isinf(escalar_duracion):
        return np.ones_like(vector_edad)
    escalar_beta = 2 * np.log(escalar_razon_qp) / escalar_duracion
    vector_exponencial = np.exp(-escalar_beta * vector_edad)
    vector_densidad = escalar_beta * (1 + escalar_razon_qp) * vector_exponencial / (1 + escalar_razon_qp * vector_exponencial) ** 2
    return np.where((vector_edad >= 0) & (vector_edad < escalar_duracion), vector_densidad, 0.0)


def factor_estacional(escalar_dia_del_anio, vector_amplitud, vector_dia_pico):
    # Factor estacional ω(τ) = exp(A / 2 · cos(2π (τ - ψ) / 365)): su logaritmo oscila entre -A / 2 y A / 2
    return np.exp(vector_amplitud / 2 * np.cos(2 * np.pi * (escalar_dia_del_anio - vector_dia_pico) / DIAS_POR_ANIO))


def muestrear_amplitudes(generador, escalar_n, vector_cuantiles, vector_probabilidades):
    # Amplitudes con la función de cuantiles que pasa por los cinco valores dados, con interpolación lineal entre ellos
    return np.interp(generador.random(escalar_n), vector_probabilidades, vector_cuantiles)


def muestrear_dias_pico(generador, escalar_n, vector_reparto):
    # El grupo de meses sale del reparto, el mes es uniforme dentro del grupo y el día es uniforme dentro del mes
    vector_reparto = np.asarray(vector_reparto, dtype=float)
    vector_grupo = generador.choice(len(MESES_POR_GRUPO_PICO), size=escalar_n, p=vector_reparto / vector_reparto.sum())
    vector_largo_grupo = np.array([len(grupo) for grupo in MESES_POR_GRUPO_PICO])
    vector_lugar = np.floor(generador.random(escalar_n) * vector_largo_grupo[vector_grupo]).astype(int)
    vector_mes = np.array([MESES_POR_GRUPO_PICO[grupo][lugar] for grupo, lugar in zip(vector_grupo, vector_lugar)], dtype=int)
    return (vector_mes + generador.random(escalar_n)) * DIAS_POR_MES


def participacion_superior(vector_participacion, escalar_fraccion=0.2):
    # Participación en la demanda de los floor(fracción · activos) productos más demandados, donde activos son los de participación positiva
    # Acepta una matriz (un día por fila) y entonces devuelve un valor por fila
    matriz = np.atleast_2d(np.asarray(vector_participacion, dtype=float))
    vector_activos = (matriz > 0).sum(axis=1)
    vector_cantidad = np.floor(escalar_fraccion * vector_activos + 1e-9).astype(int)
    matriz_acumulada = np.cumsum(-np.sort(-matriz, axis=1), axis=1)
    vector_filas = np.arange(matriz.shape[0])
    vector_superior = np.where(vector_cantidad > 0, matriz_acumulada[vector_filas, np.maximum(vector_cantidad - 1, 0)], 0.0)
    vector_suma = matriz.sum(axis=1)
    vector_resultado = np.divide(vector_superior, vector_suma, out=np.zeros(matriz.shape[0]), where=vector_suma > 0)
    return float(vector_resultado[0]) if np.ndim(vector_participacion) == 1 else vector_resultado


class Surtido:
    # Los productos del almacén: cada uno nace en un día de lanzamiento, recorre su ciclo de vida de duración L y sale al cerrarlo
    # No hay posiciones ni sucesores: el SKU de la lista de tareas es el número del producto, que no se reutiliza
    def __init__(self, vector_altura, vector_dia_lanzamiento, escalar_duracion, escalar_razon_qp, vector_amplitud, vector_dia_pico, escalar_dia_inicio_anio):
        # vector_dia_lanzamiento: día del generador en que nace cada producto, negativo (y fraccionario) si ya existía el día 0, el primero del calentamiento
        # escalar_duracion: L en días, np.inf significa sin ciclo de vida: todos los productos están activos siempre y la densidad es 1
        # escalar_dia_inicio_anio: día del generador que corresponde al 1 de enero del año evaluado, o sea los días de calentamiento
        self.vector_altura = np.asarray(vector_altura, dtype=float)
        self.vector_dia_lanzamiento = np.asarray(vector_dia_lanzamiento, dtype=float)
        self.escalar_duracion = escalar_duracion
        self.escalar_razon_qp = escalar_razon_qp
        self.vector_amplitud = np.asarray(vector_amplitud, dtype=float)
        self.vector_dia_pico = np.asarray(vector_dia_pico, dtype=float)
        self.escalar_dia_inicio_anio = escalar_dia_inicio_anio
        self.escalar_sigma_alturas = np.nan # Lo fija crear_surtido cuando las alturas salen de la curva de Hausman
        # Con lo que crear_surtido sortea los productos que nacen después: alturas posibles, nacimientos por día y días ya cubiertos
        self.vector_altura_posible = np.zeros(0)
        self.escalar_nacimientos_por_dia = 0.0
        self.escalar_dias_nacimientos = 0

    @property
    def escalar_productos(self):
        return len(self.vector_altura)

    def edades(self, escalar_dia):
        # Edad en días de cada producto; negativa si todavía no ha nacido y mayor o igual que L si ya cerró su ciclo
        return escalar_dia - self.vector_dia_lanzamiento

    def activos(self, escalar_dia):
        # Productos que han nacido y no han cerrado su ciclo ese día
        vector_edad = self.edades(escalar_dia)
        if np.isinf(self.escalar_duracion):
            return vector_edad >= 0
        return (vector_edad >= 0) & (vector_edad < self.escalar_duracion)

    def dias_cierre(self):
        # Primer día en que la edad de cada producto alcanza L: ese día sale. Sin ciclo de vida no hay cierre
        if np.isinf(self.escalar_duracion):
            return np.full(self.escalar_productos, DIA_SIN_CIERRE, dtype=np.int64)
        return np.ceil(self.vector_dia_lanzamiento + self.escalar_duracion).astype(np.int64)

    def intensidades(self, escalar_dia):
        # μ(τ) = m · g(θ) · ω(τ), el día del año cuenta desde el 1 de enero del año evaluado; vale 0 en los productos inactivos
        escalar_dia_del_anio = (escalar_dia - self.escalar_dia_inicio_anio) % DIAS_POR_ANIO
        return (self.vector_altura * densidad_bass(self.edades(escalar_dia), self.escalar_duracion, self.escalar_razon_qp)
                * factor_estacional(escalar_dia_del_anio, self.vector_amplitud, self.vector_dia_pico))

    def participaciones(self, escalar_dia):
        # ρ(τ): participación de cada producto en la demanda del día, suma 1 y es cero en los inactivos
        vector_intensidad = self.intensidades(escalar_dia)
        escalar_suma = vector_intensidad.sum()
        return vector_intensidad / escalar_suma if escalar_suma > 0 else vector_intensidad


def calibrar_sigma_alturas(matriz_base, vector_indice_altura, escalar_alturas_posibles, escalar_objetivo):
    # Busca por bisección el σ de las alturas con el que la participación diaria del 20 % de los productos activos más demandados
    # promedia el objetivo en el año. matriz_base (365, N) es g · ω de cada día del año evaluado y la concentración baja al subir σ
    # La altura de cada producto es la participación de su rango en la curva de Hausman de escalar_alturas_posibles rangos
    escalar_inferior, escalar_superior = 0.05, 1.0
    escalar_sigma = 0.5 * (escalar_inferior + escalar_superior)
    for _ in range(30):
        escalar_sigma = 0.5 * (escalar_inferior + escalar_superior)
        vector_altura = demanda_hausman(escalar_alturas_posibles, escalar_sigma)[vector_indice_altura]
        escalar_media = float(np.mean(participacion_superior(matriz_base * vector_altura)))
        if abs(escalar_media - escalar_objetivo) <= 1e-4:
            break
        if escalar_media > escalar_objetivo:
            escalar_inferior = escalar_sigma # Demasiado concentrada: hay que subir σ
        else:
            escalar_superior = escalar_sigma
    return escalar_sigma


def dias_de_nacimientos(config):
    # Días del generador en que pueden nacer productos: el calentamiento y los días que necesita la operación si avanza R recuperaciones por día
    # (la operación real avanza el doble de rápido, así que sobran días). Los productos que nazcan después ya no entran en la operación
    escalar_r = int(round(config["escalar_ciclos_por_hora"] * config["escalar_horas_por_dia"] / 2))
    return int(config["escalar_dias_calentamiento"]) + math.ceil(int(config["escalar_largo_secuencia"]) / max(escalar_r, 1))


def crear_surtido(config, generador, escalar_dias=None):
    # Arma el surtido del escenario, con nacimientos en los primeros escalar_dias días del generador. Todo el azar sale del generador, en este orden:
    # nacimientos por día, permutación de edades iniciales, rangos de altura, amplitudes y días de pico
    if config["texto_modelo_demanda"] not in ("ciclo_vida", "estatica"):
        raise ValueError(f"Modelo de demanda: '{config['texto_modelo_demanda']}' no existe. Disponibles: 'ciclo_vida', 'estatica'.")
    vector_personalizada = config["vector_demanda_sku_personalizada"]
    booleano_personalizada = vector_personalizada is not None and len(vector_personalizada) > 0
    escalar_n = len(vector_personalizada) if booleano_personalizada else int(config["escalar_numero_skus"])
    escalar_dias_calentamiento = int(config["escalar_dias_calentamiento"])
    if escalar_dias is None:
        escalar_dias = dias_de_nacimientos(config)

    if config["texto_modelo_demanda"] == "estatica":
        # Sin ciclo de vida ni estacionalidad las participaciones son constantes: la demanda estática de Hausman
        if booleano_personalizada:
            vector_altura = np.asarray(vector_personalizada, dtype=float) / np.sum(vector_personalizada)
        else:
            vector_altura = demanda_hausman(escalar_n, config["escalar_sigma_hausman"])
        surtido = Surtido(vector_altura, np.zeros(escalar_n), np.inf, config["escalar_bass_q"] / config["escalar_bass_p"],
                          np.zeros(escalar_n), np.zeros(escalar_n), escalar_dias_calentamiento)
        surtido.escalar_sigma_alturas = float(config["escalar_sigma_hausman"])
        return surtido

    escalar_duracion = config["escalar_meses_ciclo_vida"] * DIAS_POR_MES
    escalar_razon_qp = config["escalar_bass_q"] / config["escalar_bass_p"]

    # Nacimientos: el surtido medio es n y cada producto vive L días, así que nacen n / L productos por día (ley de Little) en un proceso de Poisson
    vector_nacimientos_por_dia = generador.poisson(escalar_n / escalar_duracion, size=max(int(escalar_dias), 0))
    escalar_nacidos = int(vector_nacimientos_por_dia.sum())
    vector_dia_nacidos = np.repeat(np.arange(1, len(vector_nacimientos_por_dia) + 1), vector_nacimientos_por_dia).astype(float)

    # Productos del día 0: n con edades estratificadas L / 2n, 3 L / 2n, ..., (2n - 1) L / 2n repartidas en orden aleatorio
    # El surtido arranca así en régimen: su tamaño esperado es n todos los días
    vector_edad_inicial = (generador.permutation(escalar_n) + 0.5) / escalar_n * escalar_duracion
    vector_dia_lanzamiento = np.concatenate([-vector_edad_inicial, vector_dia_nacidos])

    # Altura: cada producto toma la participación de un rango de la curva de Hausman de n rangos; los iniciales usan cada rango una vez
    # y los nacidos después sortean su rango al azar, así el surtido abierto conserva la concentración de la curva sin depender de n
    vector_indice_altura = np.concatenate([generador.permutation(escalar_n), generador.integers(0, escalar_n, size=escalar_nacidos)])
    escalar_total = escalar_n + escalar_nacidos
    vector_amplitud = muestrear_amplitudes(generador, escalar_total, config["vector_cuantiles_amplitud"], config["vector_probabilidades_amplitud"])
    vector_dia_pico = muestrear_dias_pico(generador, escalar_total, config["vector_reparto_mes_pico"])

    if booleano_personalizada:
        # Las alturas posibles son la demanda indicada, sin calibrar
        vector_altura_posible = np.asarray(vector_personalizada, dtype=float) / np.sum(vector_personalizada)
        escalar_sigma = np.nan
    elif int(np.floor(0.2 * escalar_n + 1e-9)) == 0:
        # Con menos de 5 productos el 20 % superior no contiene ninguno: no hay concentración que calibrar y se usa el σ configurado
        escalar_sigma = float(config["escalar_sigma_hausman"])
        vector_altura_posible = demanda_hausman(escalar_n, escalar_sigma)
    else:
        # σ se calibra para que la concentración media del año evaluado sea la de σ = 0,4
        surtido_unitario = Surtido(np.ones(escalar_total), vector_dia_lanzamiento, escalar_duracion, escalar_razon_qp,
                                   vector_amplitud, vector_dia_pico, escalar_dias_calentamiento)
        matriz_base = np.array([surtido_unitario.intensidades(escalar_dias_calentamiento + escalar_dia) for escalar_dia in range(DIAS_POR_ANIO)])
        escalar_sigma = calibrar_sigma_alturas(matriz_base, vector_indice_altura, escalar_n, 0.2 ** config["escalar_sigma_hausman"])
        vector_altura_posible = demanda_hausman(escalar_n, escalar_sigma)

    surtido = Surtido(vector_altura_posible[vector_indice_altura], vector_dia_lanzamiento, escalar_duracion, escalar_razon_qp,
                      vector_amplitud, vector_dia_pico, escalar_dias_calentamiento)
    surtido.escalar_sigma_alturas = escalar_sigma
    surtido.vector_altura_posible = vector_altura_posible
    surtido.escalar_nacimientos_por_dia = escalar_n / escalar_duracion
    surtido.escalar_dias_nacimientos = int(escalar_dias)
    return surtido


def ampliar_surtido(surtido, config, generador, escalar_dias):
    # Extiende los nacimientos hasta escalar_dias días del generador, con las mismas reglas y el σ ya calibrado
    # Se usa solo si la operación avanza más lento de lo previsto (por ejemplo en estanterías pequeñas, donde parte de la demanda queda sin atender)
    # Devuelve cuántos productos se agregaron, que reciben los números de SKU siguientes
    escalar_dias_nuevos = int(escalar_dias) - surtido.escalar_dias_nacimientos
    if escalar_dias_nuevos <= 0:
        return 0
    escalar_dia_anterior = surtido.escalar_dias_nacimientos
    surtido.escalar_dias_nacimientos = int(escalar_dias)
    if surtido.escalar_nacimientos_por_dia <= 0:
        return 0
    vector_nacimientos_por_dia = generador.poisson(surtido.escalar_nacimientos_por_dia, size=escalar_dias_nuevos)
    escalar_nacidos = int(vector_nacimientos_por_dia.sum())
    vector_dia_nacidos = np.repeat(np.arange(escalar_dia_anterior + 1, int(escalar_dias) + 1), vector_nacimientos_por_dia).astype(float)
    vector_indice_altura = generador.integers(0, len(surtido.vector_altura_posible), size=escalar_nacidos)
    vector_amplitud = muestrear_amplitudes(generador, escalar_nacidos, config["vector_cuantiles_amplitud"], config["vector_probabilidades_amplitud"])
    vector_dia_pico = muestrear_dias_pico(generador, escalar_nacidos, config["vector_reparto_mes_pico"])
    surtido.vector_altura = np.concatenate([surtido.vector_altura, surtido.vector_altura_posible[vector_indice_altura]])
    surtido.vector_dia_lanzamiento = np.concatenate([surtido.vector_dia_lanzamiento, vector_dia_nacidos])
    surtido.vector_amplitud = np.concatenate([surtido.vector_amplitud, vector_amplitud])
    surtido.vector_dia_pico = np.concatenate([surtido.vector_dia_pico, vector_dia_pico])
    return escalar_nacidos
