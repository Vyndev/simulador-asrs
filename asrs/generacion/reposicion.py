"""Reposición con lotes EOQ ligada al inventario de cada producto: genera el escenario de demanda (listas de llenado y de operación) del modelo de suministro y demanda."""
import math

import numpy as np

from asrs.generacion.ciclo_vida import crear_cartera, participacion_superior
from asrs.generacion.escenario import EscenarioDemanda

DIAS_SIN_AVANCE = 730 # Días seguidos de operación sin tareas nuevas tras los que el generador se declara bloqueado


def recuperaciones_diarias(config):
    # R: recuperaciones de demanda de un día, la mitad de las tareas del día con la mezcla de 50 % de almacenamientos y 50 % de recuperaciones
    return int(round(config["escalar_ciclos_por_hora"] * config["escalar_horas_por_dia"] / 2))


def lotes_eoq(vector_demanda_esperada, escalar_kappa):
    # Q = max(1, redondeo(κ √ν)): el lote económico cuando la razón entre el costo de pedir y el de mantener es común a todos los productos
    vector_raiz = np.sqrt(np.asarray(vector_demanda_esperada, dtype=float))
    return np.maximum(1, np.rint(escalar_kappa * vector_raiz)).astype(int)


def _inventario_esperado(vector_raiz, escalar_kappa):
    # Σ (Q + 1) / 2: con punto de pedido 0, el inventario de un producto recorre Q, Q - 1, ..., 1
    return float((np.maximum(1, np.rint(escalar_kappa * vector_raiz)) + 1).sum() / 2)


def calibrar_kappa(vector_demanda_esperada, escalar_inventario_objetivo):
    # Bisección de κ para que el inventario esperado quede lo más cerca posible del objetivo
    # El inventario esperado es una función escalonada y no decreciente de κ, así que se busca el salto que cruza el objetivo
    vector_raiz = np.sqrt(np.asarray(vector_demanda_esperada, dtype=float))
    if vector_raiz.max(initial=0.0) <= 0 or _inventario_esperado(vector_raiz, 0.0) >= escalar_inventario_objetivo:
        return 0.0 # Con lotes de una unidad ya se alcanza el objetivo
    escalar_inferior, escalar_superior = 0.0, 1.0
    while _inventario_esperado(vector_raiz, escalar_superior) < escalar_inventario_objetivo:
        escalar_inferior, escalar_superior = escalar_superior, 2 * escalar_superior
    # Se mantiene inventario(inferior) < objetivo <= inventario(superior)
    for _ in range(60):
        escalar_medio = 0.5 * (escalar_inferior + escalar_superior)
        if _inventario_esperado(vector_raiz, escalar_medio) < escalar_inventario_objetivo:
            escalar_inferior = escalar_medio
        else:
            escalar_superior = escalar_medio
        if escalar_superior - escalar_inferior <= 1e-12 * escalar_superior:
            break
    escalar_error_inferior = escalar_inventario_objetivo - _inventario_esperado(vector_raiz, escalar_inferior)
    escalar_error_superior = _inventario_esperado(vector_raiz, escalar_superior) - escalar_inventario_objetivo
    return escalar_inferior if escalar_error_inferior < escalar_error_superior else escalar_superior


class _ListaCompleta(Exception):
    # La lista de operación llegó a N tareas: el generador se detiene, aunque sea a mitad de día
    pass


class _GeneradorEscenario:
    # Lleva la demanda, los lotes y el inventario de cada posición del surtido día por día, desde el primer día del calentamiento
    # El inventario se lleva sobre la lista y no sobre las celdas, por eso la lista no depende de la política
    def __init__(self, config, generador):
        self.generador = generador
        self.cartera = crear_cartera(config, generador) # La cartera es lo primero que sale del generador
        self.escalar_n = len(self.cartera.vector_altura)
        escalar_celdas = int(np.prod(config["vector_configuracion_estanteria"]))
        self.escalar_objetivo = config["escalar_nivel_ocupacion"] * escalar_celdas # Inventario esperado φ F C K
        self.escalar_techo = int(np.floor(config["escalar_techo_ocupacion"] * escalar_celdas + 1e-9)) # floor(φ̄ F C K)
        self.escalar_r = recuperaciones_diarias(config)
        if self.escalar_r < 1:
            raise ValueError("Recuperaciones por día: los ciclos por hora y las horas por día dan menos de una recuperación diaria.")
        self.escalar_largo = int(config["escalar_largo_secuencia"])
        self.escalar_calentamiento = int(config["escalar_dias_calentamiento"])
        self.escalar_ventana = int(config["escalar_dias_ventana_popularidad"])
        self.booleano_ciclo = bool(np.isfinite(self.cartera.escalar_duracion)) # Con el modelo estático no hay cierres ni lanzamientos
        # Días que la matriz de demanda cubre más allá del día en curso: hasta el próximo cierre de cualquier posición
        self.escalar_margen = (math.ceil(self.cartera.escalar_duracion) if self.booleano_ciclo else 0) + 1
        # Desfase uniforme por posición para el redondeo acumulado de la demanda
        self.vector_desfase = generador.random(self.escalar_n)
        self.preparar_demanda(self.escalar_calentamiento + math.ceil(self.escalar_largo / self.escalar_r) + self.escalar_margen)
        # Tuplas de cada tarea, armadas una vez: el SKU es el número de la posición
        self.lista_tupla_almacenar = [(0, escalar_posicion + 1) for escalar_posicion in range(self.escalar_n)]
        self.lista_tupla_recuperar = [(1, escalar_posicion + 1) for escalar_posicion in range(self.escalar_n)]

    def preparar_demanda(self, escalar_dias):
        # Precalcula, para los días 0 a escalar_dias - 1:
        #   ν = R ρ, la demanda esperada de cada posición;
        #   su acumulado, cuya fila d es la demanda esperada de los días 0 a d - 1 (sirve para la demanda restante hasta el cierre);
        #   las recuperaciones de cada día, floor(E(d) + desfase) - floor(E(d - 1) + desfase), exactas en el acumulado de cada posición;
        #   la generación de cada posición, que aumenta el día en que cierra su ciclo
        # No usa el generador aleatorio, así que ampliarla no cambia el escenario
        self.escalar_dias = escalar_dias
        self.matriz_demanda = self.escalar_r * np.array([self.cartera.participaciones(escalar_dia) for escalar_dia in range(escalar_dias)])
        self.matriz_acumulada = np.vstack([np.zeros(self.escalar_n), np.cumsum(self.matriz_demanda, axis=0)])
        self.matriz_recuperaciones = np.diff(np.floor(self.matriz_acumulada + self.vector_desfase).astype(np.int64), axis=0)
        self.matriz_generacion = np.array([self.cartera.generaciones(escalar_dia) for escalar_dia in range(escalar_dias)])

    def proximo_cierre(self, escalar_dia, escalar_posicion):
        # Primer día posterior a escalar_dia en que la generación de la posición aumenta; si no está en la matriz, su último día
        vector_columna = self.matriz_generacion[escalar_dia + 1:, escalar_posicion] > self.matriz_generacion[escalar_dia, escalar_posicion]
        if vector_columna.any():
            return escalar_dia + 1 + int(np.argmax(vector_columna))
        return self.escalar_dias

    def lotes_del_dia(self, escalar_dia):
        # κ(d) y Q(d) al empezar el día. Un lote mayor que el techo nunca cabría, así que se acota a él (solo pasa con estanterías diminutas)
        vector_demanda = self.matriz_demanda[escalar_dia]
        vector_lote = lotes_eoq(vector_demanda, calibrar_kappa(vector_demanda, self.escalar_objetivo))
        return np.minimum(vector_lote, max(1, self.escalar_techo)).tolist()

    def emitir_bloque(self, escalar_actividad, escalar_posicion, escalar_cantidad, booleano_retiro=False):
        # Emite escalar_cantidad tareas seguidas de la misma posición, cortando si la lista de operación llega a N
        escalar_cantidad = int(min(escalar_cantidad, self.escalar_limite - len(self.lista_tareas)))
        escalar_inicio = len(self.lista_tareas)
        if escalar_actividad == 0:
            self.lista_tareas.extend([self.lista_tupla_almacenar[escalar_posicion]] * escalar_cantidad)
            self.lista_ocupacion.extend(range(self.escalar_total + 1, self.escalar_total + escalar_cantidad + 1))
            self.escalar_total += escalar_cantidad
            self.vector_inventario[escalar_posicion] += escalar_cantidad
        else:
            self.lista_tareas.extend([self.lista_tupla_recuperar[escalar_posicion]] * escalar_cantidad)
            self.lista_ocupacion.extend(range(self.escalar_total - 1, self.escalar_total - escalar_cantidad - 1, -1))
            self.escalar_total -= escalar_cantidad
            self.vector_inventario[escalar_posicion] -= escalar_cantidad
            if booleano_retiro:
                self.conjunto_retiros.update(range(escalar_inicio, escalar_inicio + escalar_cantidad))
        self.lista_dias.extend([self.escalar_dia_tarea] * escalar_cantidad)
        if len(self.lista_tareas) >= self.escalar_limite:
            raise _ListaCompleta

    def insertar_lote(self, escalar_posicion, escalar_lote):
        # (f) El lote entra con sus unidades seguidas si cabe bajo el techo; si no, queda pendiente hasta que haya espacio
        if self.escalar_total + escalar_lote <= self.escalar_techo:
            self.emitir_bloque(0, escalar_posicion, escalar_lote)
        else:
            self.lista_pendientes.append((escalar_posicion, escalar_lote))
            self.escalar_lotes_aplazados += 1

    def intentar_pendientes(self):
        # Los lotes pendientes se revisan en orden de llegada y entra cada uno que quepa bajo el techo
        lista_revisar, self.lista_pendientes = self.lista_pendientes, []
        for escalar_posicion, escalar_lote in lista_revisar:
            if self.escalar_total + escalar_lote <= self.escalar_techo:
                self.emitir_bloque(0, escalar_posicion, escalar_lote)
            else:
                self.lista_pendientes.append((escalar_posicion, escalar_lote))

    def reponer(self, escalar_dia, escalar_posicion):
        # La posición quedó sin unidades: se pide un lote, con el corte de compras de Kurawarwala y Matsuo (1996)
        # Solo se pide Q si la demanda esperada que le queda al producto hasta el cierre, menos lo ya emitido hoy, es al menos Q;
        # si no, un lote final del tamaño de esa demanda redondeada hacia arriba, y ninguno si no queda demanda
        escalar_lote = self.vector_lote[escalar_posicion]
        if self.booleano_ciclo: # En el modelo estático la demanda restante es infinita y el lote siempre es Q
            escalar_cierre = self.vector_proximo_cierre[escalar_posicion]
            escalar_restante = (self.matriz_acumulada[escalar_cierre, escalar_posicion] - self.matriz_acumulada[escalar_dia, escalar_posicion]
                                - self.vector_hoy[escalar_posicion])
            if escalar_restante < escalar_lote:
                if escalar_restante <= 0:
                    return
                escalar_lote = math.ceil(escalar_restante)
        self.insertar_lote(escalar_posicion, escalar_lote)

    def cerrar_ciclos(self, escalar_dia, lista_cierre):
        # (c) Cierres del día, en orden de posición: las unidades que quedan salen como retiros seguidos, la posición queda en cero
        # y entra el primer lote del sucesor. Los lotes pendientes del producto que sale se cancelan antes, para que ninguno entre y salga enseguida
        # (un producto con lote pendiente no tiene unidades, así que no tiene retiros)
        conjunto_cierre = set(lista_cierre)
        self.lista_pendientes = [(escalar_posicion, escalar_lote) for escalar_posicion, escalar_lote in self.lista_pendientes
                                 if escalar_posicion not in conjunto_cierre]
        for escalar_posicion in lista_cierre:
            if self.vector_inventario[escalar_posicion] > 0:
                self.emitir_bloque(1, escalar_posicion, self.vector_inventario[escalar_posicion], booleano_retiro=True)
            self.vector_proximo_cierre[escalar_posicion] = self.proximo_cierre(escalar_dia, escalar_posicion)
            self.escalar_lanzamientos += 1
            self.insertar_lote(escalar_posicion, self.vector_lote[escalar_posicion])
            if self.lista_pendientes: # El retiro liberó espacio
                self.intentar_pendientes()

    def recuperar_dia(self, escalar_dia):
        # (e) Recuperaciones de demanda del día, en orden aleatorio. Es el ciclo más repetido del generador: usa variables locales
        vector_orden = self.generador.permutation(np.repeat(np.arange(self.escalar_n), self.matriz_recuperaciones[escalar_dia])).tolist()
        lista_tareas, lista_dias, lista_ocupacion = self.lista_tareas, self.lista_dias, self.lista_ocupacion
        vector_inventario, vector_hoy, lista_tupla = self.vector_inventario, self.vector_hoy, self.lista_tupla_recuperar
        escalar_dia_tarea, escalar_limite = self.escalar_dia_tarea, self.escalar_limite
        for escalar_posicion in vector_orden:
            if vector_inventario[escalar_posicion] == 0:
                self.escalar_no_atendida += 1 # Sin unidades y con el lote aplazado o sin lote: la recuperación no se emite
                continue
            vector_inventario[escalar_posicion] -= 1
            vector_hoy[escalar_posicion] += 1
            self.escalar_total -= 1
            lista_tareas.append(lista_tupla[escalar_posicion])
            lista_dias.append(escalar_dia_tarea)
            lista_ocupacion.append(self.escalar_total)
            if len(lista_tareas) >= escalar_limite:
                raise _ListaCompleta
            if vector_inventario[escalar_posicion] == 0:
                self.reponer(escalar_dia, escalar_posicion)
            if self.lista_pendientes:
                self.intentar_pendientes()

    def empezar_operacion(self):
        # (a) Fin del calentamiento: el inventario define la lista de llenado, en orden aleatorio, y desde aquí las tareas van a la operación
        self.vector_inventario_inicial = np.array(self.vector_inventario, dtype=int)
        vector_skus = self.generador.permutation(np.repeat(np.arange(1, self.escalar_n + 1), self.vector_inventario_inicial)).tolist()
        self.lista_llenado = [(0, escalar_sku) for escalar_sku in vector_skus]
        self.lista_tareas, self.lista_dias, self.lista_ocupacion = [], [], []
        self.conjunto_retiros = set()
        self.escalar_no_atendida = self.escalar_lotes_aplazados = self.escalar_lanzamientos = 0
        self.escalar_limite = self.escalar_largo

    def generar(self):
        # Día 0: cada producto parte con la mitad de su lote, redondeada hacia arriba
        self.vector_lote = self.lotes_del_dia(0)
        self.vector_inventario = [(escalar_lote + 1) // 2 for escalar_lote in self.vector_lote]
        self.escalar_total = sum(self.vector_inventario)
        self.lista_pendientes = []
        self.vector_proximo_cierre = ([self.proximo_cierre(0, escalar_posicion) for escalar_posicion in range(self.escalar_n)]
                                      if self.booleano_ciclo else [math.inf] * self.escalar_n)
        # Recuperaciones de demanda de cada posición en los últimos W días, en un arreglo circular por día
        matriz_historia = np.zeros((max(self.escalar_ventana, 1), self.escalar_n), dtype=np.int64)
        lista_popularidad = []
        # Durante el calentamiento las tareas se llevan igual, sin límite, y se descartan al empezar la operación
        self.lista_tareas, self.lista_dias, self.lista_ocupacion = [], [], []
        self.conjunto_retiros = set()
        self.escalar_no_atendida = self.escalar_lotes_aplazados = self.escalar_lanzamientos = 0
        self.escalar_limite = math.inf
        self.vector_inventario_inicial = None
        escalar_dias_sin_avance = 0
        escalar_dia = 0
        try:
            while True:
                if escalar_dia + self.escalar_margen >= self.escalar_dias:
                    self.preparar_demanda(2 * self.escalar_dias) # Solo si la operación avanza más lento que R recuperaciones por día
                if escalar_dia == self.escalar_calentamiento:
                    self.empezar_operacion()
                booleano_operacion = escalar_dia >= self.escalar_calentamiento
                if booleano_operacion and len(self.lista_tareas) >= self.escalar_largo:
                    break
                escalar_largo_previo = len(self.lista_tareas)
                # (b) κ(d) y Q(d)
                self.vector_lote = self.lotes_del_dia(escalar_dia)
                # Posiciones cuyo producto cierra el ciclo hoy: su historia de popularidad se reinicia antes de anotar la fila del día
                lista_cierre = []
                if self.booleano_ciclo and escalar_dia > 0:
                    lista_cierre = np.flatnonzero(self.matriz_generacion[escalar_dia] > self.matriz_generacion[escalar_dia - 1]).tolist()
                    matriz_historia[:, lista_cierre] = 0
                # (d) Popularidad móvil al empezar el día: días [max(lanzamiento, d - W), d - 1]. Ningún paso del cierre la cambia
                # salvo el reinicio de arriba, así que anotarla aquí da la misma fila y queda anotada aunque la lista termine en el cierre
                if booleano_operacion:
                    lista_popularidad.append(matriz_historia.sum(axis=0) if self.escalar_ventana > 0 else np.zeros(self.escalar_n, dtype=np.int64))
                self.vector_hoy = [0] * self.escalar_n
                self.escalar_dia_tarea = escalar_dia - self.escalar_calentamiento
                # (c) Cierres y (e) recuperaciones del día
                self.cerrar_ciclos(escalar_dia, lista_cierre)
                self.recuperar_dia(escalar_dia)
                # Fin del día: sus recuperaciones de demanda reemplazan las del día d - W en el arreglo circular
                matriz_historia[escalar_dia % matriz_historia.shape[0]] = self.vector_hoy
                if booleano_operacion:
                    escalar_dias_sin_avance = escalar_dias_sin_avance + 1 if len(self.lista_tareas) == escalar_largo_previo else 0
                    if escalar_dias_sin_avance >= DIAS_SIN_AVANCE:
                        raise RuntimeError("Escenario de demanda: la lista de operación no avanza, ningún lote cabe bajo el techo.")
                escalar_dia += 1
        except _ListaCompleta:
            pass
        return self.escenario(lista_popularidad)

    def escenario(self, lista_popularidad):
        # Arma el escenario y su resumen. La ocupación se mide con el total después de cada tarea de la operación
        escalar_dias_operacion = len(lista_popularidad)
        matriz_popularidad = (np.array(lista_popularidad, dtype=np.int64) if lista_popularidad
                              else np.zeros((0, self.escalar_n), dtype=np.int64))
        vector_ocupacion = np.array(self.lista_ocupacion if self.lista_ocupacion else [len(self.lista_llenado)])
        if escalar_dias_operacion:
            matriz_dias = self.matriz_demanda[self.escalar_calentamiento:self.escalar_calentamiento + escalar_dias_operacion]
            escalar_top20 = float(np.mean(participacion_superior(matriz_dias)))
        else:
            escalar_top20 = float("nan")
        dic_resumen = {
            "demanda_no_atendida": int(self.escalar_no_atendida),
            "lotes_aplazados": int(self.escalar_lotes_aplazados),
            "retiros": int(len(self.conjunto_retiros)), # Tareas de retiro de fin de ciclo en la operación
            "lanzamientos": int(self.escalar_lanzamientos),
            "ocupacion_media": float(vector_ocupacion.mean()),
            "ocupacion_minima": int(vector_ocupacion.min()),
            "ocupacion_maxima": int(vector_ocupacion.max()),
            "participacion_top20_media": escalar_top20,
            "sigma_alturas": float(self.cartera.escalar_sigma_alturas),
            "dias_operacion": escalar_dias_operacion,
        }
        return EscenarioDemanda(self.lista_llenado, self.lista_tareas, self.lista_dias, self.conjunto_retiros,
                                matriz_popularidad, self.vector_inventario_inicial, dic_resumen)


def generar_escenario(config, generador):
    # Genera la lista de llenado y la de operación con el modelo de suministro y demanda. Todo el azar sale de generador
    return _GeneradorEscenario(config, generador).generar()
