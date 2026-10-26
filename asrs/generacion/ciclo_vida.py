"""Cartera de productos con ciclo de vida de Bass y estacionalidad: intensidad y participación de la demanda de cada posición del surtido."""
import numpy as np

from asrs.generacion.probabilidades import demanda_hausman

DIAS_POR_ANIO = 365
DIAS_POR_MES = 365 / 12 # Un mes son 365/12 días, unos 30,42
# Meses de cada grupo de pico estacional, numerados de 0 (enero) a 11: diciembre a febrero, junio a agosto y el resto
MESES_POR_GRUPO_PICO = ((11, 0, 1), (5, 6, 7), (2, 3, 4, 8, 9, 10))


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
    # Factor estacional ω(τ) = exp(A / 2 · cos(2π (τ - φ) / 365)): su logaritmo oscila entre -A / 2 y A / 2
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
    vector_mes = np.array([MESES_POR_GRUPO_PICO[grupo][lugar] for grupo, lugar in zip(vector_grupo, vector_lugar)])
    return (vector_mes + generador.random(escalar_n)) * DIAS_POR_MES


def participacion_superior(vector_participacion, escalar_fraccion=0.2):
    # Participación en la demanda de los floor(fracción · n) productos más demandados
    # Acepta una matriz (un día por fila) y entonces devuelve un valor por fila
    matriz = np.atleast_2d(np.asarray(vector_participacion, dtype=float))
    escalar_cantidad = int(np.floor(escalar_fraccion * matriz.shape[1] + 1e-9))
    matriz_ordenada = -np.sort(-matriz, axis=1)
    vector_resultado = matriz_ordenada[:, :escalar_cantidad].sum(axis=1) / matriz.sum(axis=1)
    return float(vector_resultado[0]) if np.ndim(vector_participacion) == 1 else vector_resultado


class CarteraProductos:
    # Las posiciones del surtido: cada una tiene siempre un producto activo y al cerrar su ciclo lanza un sucesor que hereda la altura y el perfil estacional
    def __init__(self, vector_altura, vector_edad_inicial, escalar_duracion, escalar_razon_qp, vector_amplitud, vector_dia_pico, escalar_dia_inicio_anio):
        # vector_edad_inicial: edad de cada producto en el día 0 del generador, el primero del calentamiento
        # escalar_duracion: L en días, np.inf significa sin ciclo de vida (la edad no avanza y la densidad es 1)
        # escalar_dia_inicio_anio: día del generador que corresponde al 1 de enero del año evaluado, o sea los días de calentamiento
        self.vector_altura = np.asarray(vector_altura, dtype=float)
        self.vector_edad_inicial = np.asarray(vector_edad_inicial, dtype=float)
        self.escalar_duracion = escalar_duracion
        self.escalar_razon_qp = escalar_razon_qp
        self.vector_amplitud = np.asarray(vector_amplitud, dtype=float)
        self.vector_dia_pico = np.asarray(vector_dia_pico, dtype=float)
        self.escalar_dia_inicio_anio = escalar_dia_inicio_anio
        self.escalar_sigma_alturas = np.nan # Lo fija crear_cartera cuando las alturas salen de la curva de Hausman

    def edades(self, escalar_dia):
        # Edad en días del producto activo de cada posición, el ciclo se cierra cuando la edad alcanza L
        if np.isinf(self.escalar_duracion):
            return self.vector_edad_inicial.copy()
        return np.mod(self.vector_edad_inicial + escalar_dia, self.escalar_duracion)

    def generaciones(self, escalar_dia):
        # Ciclos completos que ha cerrado cada posición desde el día 0, o sea cuántos sucesores se han lanzado
        if np.isinf(self.escalar_duracion):
            return np.zeros(len(self.vector_altura), dtype=int)
        return np.floor((self.vector_edad_inicial + escalar_dia) / self.escalar_duracion).astype(int)

    def intensidades(self, escalar_dia):
        # μ(τ) = m · g(θ) · ω(τ), el día del año cuenta desde el 1 de enero del año evaluado
        escalar_dia_del_anio = (escalar_dia - self.escalar_dia_inicio_anio) % DIAS_POR_ANIO
        return (self.vector_altura * densidad_bass(self.edades(escalar_dia), self.escalar_duracion, self.escalar_razon_qp)
                * factor_estacional(escalar_dia_del_anio, self.vector_amplitud, self.vector_dia_pico))

    def participaciones(self, escalar_dia):
        # ρ(τ): participación de cada posición en la demanda del día, suma 1
        vector_intensidad = self.intensidades(escalar_dia)
        return vector_intensidad / vector_intensidad.sum()


def calibrar_sigma_alturas(matriz_base, vector_orden_alturas, escalar_objetivo):
    # Busca por bisección el σ de las alturas con el que la participación diaria del 20 % de los productos más demandados
    # promedia el objetivo en el año. matriz_base (365, n) es g · ω de cada día del año evaluado y la concentración baja al subir σ
    escalar_inferior, escalar_superior = 0.05, 1.0
    escalar_sigma = 0.5 * (escalar_inferior + escalar_superior)
    for _ in range(30):
        escalar_sigma = 0.5 * (escalar_inferior + escalar_superior)
        vector_altura = demanda_hausman(matriz_base.shape[1], escalar_sigma)[vector_orden_alturas]
        escalar_media = float(np.mean(participacion_superior(matriz_base * vector_altura)))
        if abs(escalar_media - escalar_objetivo) <= 1e-4:
            break
        if escalar_media > escalar_objetivo:
            escalar_inferior = escalar_sigma # Demasiado concentrada: hay que subir σ
        else:
            escalar_superior = escalar_sigma
    return escalar_sigma


def crear_cartera(config, generador):
    # Arma la cartera del escenario. Todo el azar sale del generador, en este orden: permutación de edades, amplitudes, días de pico y permutación de alturas
    if config["texto_modelo_demanda"] not in ("ciclo_vida", "estatica"):
        raise ValueError(f"Modelo de demanda: '{config['texto_modelo_demanda']}' no existe. Disponibles: 'ciclo_vida', 'estatica'.")
    vector_personalizada = config["vector_demanda_sku_personalizada"]
    booleano_personalizada = vector_personalizada is not None and len(vector_personalizada) > 0
    escalar_n = len(vector_personalizada) if booleano_personalizada else int(config["escalar_numero_skus"])
    escalar_dias_calentamiento = int(config["escalar_dias_calentamiento"])

    if config["texto_modelo_demanda"] == "estatica":
        # Sin ciclo de vida ni estacionalidad las participaciones son constantes: la demanda estática de Hausman
        if booleano_personalizada:
            vector_altura = np.asarray(vector_personalizada, dtype=float) / np.sum(vector_personalizada)
        else:
            vector_altura = demanda_hausman(escalar_n, config["escalar_sigma_hausman"])
        cartera = CarteraProductos(vector_altura, np.zeros(escalar_n), np.inf, config["escalar_bass_q"] / config["escalar_bass_p"],
                                   np.zeros(escalar_n), np.zeros(escalar_n), escalar_dias_calentamiento)
        cartera.escalar_sigma_alturas = float(config["escalar_sigma_hausman"])
        return cartera

    escalar_duracion = config["escalar_meses_ciclo_vida"] * DIAS_POR_MES
    escalar_razon_qp = config["escalar_bass_q"] / config["escalar_bass_p"]
    # Edades estratificadas: L / 2n, 3 L / 2n, ..., (2n - 1) L / 2n repartidas en orden aleatorio, para que el surtido arranque en régimen
    vector_edad_inicial = (generador.permutation(escalar_n) + 0.5) / escalar_n * escalar_duracion
    vector_amplitud = muestrear_amplitudes(generador, escalar_n, config["vector_cuantiles_amplitud"], config["vector_probabilidades_amplitud"])
    vector_dia_pico = muestrear_dias_pico(generador, escalar_n, config["vector_reparto_mes_pico"])

    if booleano_personalizada:
        # Las alturas son la demanda indicada, en orden de SKU y sin calibrar
        vector_altura = np.asarray(vector_personalizada, dtype=float) / np.sum(vector_personalizada)
        escalar_sigma = np.nan
    else:
        # Alturas de Hausman en orden aleatorio, independiente de la edad. σ se calibra para que la concentración media del año evaluado sea la de σ = 0,4
        vector_orden_alturas = generador.permutation(escalar_n)
        if int(np.floor(0.2 * escalar_n + 1e-9)) == 0:
            # Con menos de 5 posiciones el 20 % superior no contiene ninguna: no hay concentración que calibrar y se usa el σ configurado
            escalar_sigma = float(config["escalar_sigma_hausman"])
        else:
            cartera_unitaria = CarteraProductos(np.ones(escalar_n), vector_edad_inicial, escalar_duracion, escalar_razon_qp,
                                                vector_amplitud, vector_dia_pico, escalar_dias_calentamiento)
            matriz_base = np.array([cartera_unitaria.intensidades(escalar_dias_calentamiento + escalar_dia) for escalar_dia in range(DIAS_POR_ANIO)])
            escalar_sigma = calibrar_sigma_alturas(matriz_base, vector_orden_alturas, 0.2 ** config["escalar_sigma_hausman"])
        vector_altura = demanda_hausman(escalar_n, escalar_sigma)[vector_orden_alturas]

    cartera = CarteraProductos(vector_altura, vector_edad_inicial, escalar_duracion, escalar_razon_qp, vector_amplitud, vector_dia_pico, escalar_dias_calentamiento)
    cartera.escalar_sigma_alturas = escalar_sigma
    return cartera
