"""Grúa que se desplaza entre el punto E/S y la estantería."""
import numpy as np

from asrs.cinematica import tiempo_chebyshev_xy
from asrs.metricas import Metricas


class Grua:
    def __init__(self, vector_posicion_reposo, vector_velocidad, tiempo_cargadescarga, vector_aceleracion=None, vector_deceleracion=None):
        # Inicializa la grúa con una posición de reposo y una velocidad
        # La posición actual de la grúa se establece en la posición de reposo
        # La grúa inicialmente no tiene ninguna carga ni está en ninguna estantería o punto E/S
        # Las posiciones se manejan como flotantes y como copias independientes, así se evitan truncados y modificaciones cruzadas
        self.vector_posicion_resposo = np.array(vector_posicion_reposo, dtype=float)
        self.vector_posicion_actual = np.copy(self.vector_posicion_resposo)
        self.vector_velocidad = np.array(vector_velocidad, dtype=float)
        # Aceleración y deceleración de cada eje (filas, columnas, canales), 0 significa sin límite
        self.vector_aceleracion = np.zeros(3) if vector_aceleracion is None else np.array(vector_aceleracion, dtype=float)
        self.vector_deceleracion = np.zeros(3) if vector_deceleracion is None else np.array(vector_deceleracion, dtype=float)
        self.vector_carga = []
        self.instancia_estanteria_actual = 0
        self.instancia_puntoES_actual = 0
        self.tiempo_cargadescarga = tiempo_cargadescarga
        self.intancia_metricas = Metricas()

        # Reloj de la grúa y registro opcional de eventos, sirven para visualizar la simulación
        # El registro está desactivado (None) salvo que se asigne una lista a 'vector_eventos'
        self.escalar_reloj = 0.0
        self.vector_eventos = None

    # Métodos de registro de eventos

    def registrarEvento(self, txt_tipo, *vector_datos):
        # Agrega un evento al registro con el instante actual del reloj, si el registro está activo
        if self.vector_eventos is not None:
            self.vector_eventos.append([round(self.escalar_reloj, 4), txt_tipo, *vector_datos])

    def _tiempo(self, vector_posicion_inicial, vector_posicion_final):
        # Tiempo de un desplazamiento con la velocidad, la aceleración y la deceleración de la grúa
        return self.intancia_metricas.tiempoDezplamiento(vector_posicion_inicial, vector_posicion_final, self.vector_velocidad,
                                                         self.vector_aceleracion, self.vector_deceleracion)

    def _enES(self):
        # Indica si la grúa está en el punto E/S (o en reposo) y no dentro del recorrido de una estantería
        return self.instancia_estanteria_actual == 0

    def _registrarMovimiento(self, vector_posicion_inicial, vector_posicion_final, escalar_tiempo, booleano_en_es_inicial, booleano_en_es_final):
        # Registra un desplazamiento (con el tiempo xy aparte, porque los ejes xy son simultáneos y z es secuencial) y avanza el reloj
        if self.vector_eventos is not None:
            escalar_tiempo_xy = float(tiempo_chebyshev_xy(np.abs(vector_posicion_final[:2] - vector_posicion_inicial[:2]), self.vector_velocidad[:2],
                                                          self.vector_aceleracion[:2], self.vector_deceleracion[:2]))
            self.registrarEvento('M', round(escalar_tiempo, 4),
                                 round(float(vector_posicion_inicial[0]), 4), round(float(vector_posicion_inicial[1]), 4),
                                 round(float(vector_posicion_final[0]), 4), round(float(vector_posicion_final[1]), 4),
                                 round(escalar_tiempo_xy, 4),
                                 round(float(vector_posicion_inicial[2]), 4), round(float(vector_posicion_final[2]), 4),
                                 int(booleano_en_es_inicial), int(booleano_en_es_final))
        self.escalar_reloj += escalar_tiempo
        return escalar_tiempo

    def _registrarCarga(self, txt_tipo, *vector_datos):
        # Registra una operación de carga o descarga y avanza el reloj
        self.registrarEvento(txt_tipo, *vector_datos)
        self.escalar_reloj += self.tiempo_cargadescarga
        return self.tiempo_cargadescarga

    def _aplanar(self):
        # La grúa se mueve a la altura del riel (posición z de reposo), devuelve el tiempo sin registrarlo
        # Se conserva una copia de la posición inicial, de lo contrario se modificaría en el sitio y el tiempo siempre sería 0
        vector_posicion_inicial = np.copy(self.vector_posicion_actual)
        self.vector_posicion_actual[2] = self.vector_posicion_resposo[2]
        vector_posicion_final = np.copy(self.vector_posicion_actual)

        return vector_posicion_inicial, vector_posicion_final, self._tiempo(vector_posicion_inicial, vector_posicion_final)

    # Métodos para interactuar con la estantería

    def entrarAlPasillo(self, instancia_estanteria):
        # La grúa pasa a operar en la estantería dada, a la altura del riel y desde donde está
        # No pasa por la esquina de la estantería: desde el punto E/S o desde la última celda atendida va directo a la siguiente
        # Si ya estaba en el pasillo (por ejemplo, después de reubicar) el movimiento es nulo
        vector_posicion_inicial = np.copy(self.vector_posicion_actual)
        booleano_en_es_inicial = self._enES()

        self.instancia_puntoES_actual = 0
        self.instancia_estanteria_actual = instancia_estanteria
        self._aplanar()
        vector_posicion_final = np.copy(self.vector_posicion_actual)

        return self._registrarMovimiento(vector_posicion_inicial, vector_posicion_final,
                                         self._tiempo(vector_posicion_inicial, vector_posicion_final),
                                         booleano_en_es_inicial, False)

    def moverEnEstanteria(self, vector_indice_destino):
        # La grúa se mueve dentro de la estantería a una posición dada
        # La posición actual de la grúa se actualiza a la nueva posición
        vector_posicion_inicial = np.copy(self.vector_posicion_actual)
        vector_posicion_final = np.array(self.instancia_estanteria_actual.vector_posicion, dtype=float) + (np.array(vector_indice_destino) * np.copy(self.instancia_estanteria_actual.vector_cotas)) + (
            np.copy(self.instancia_estanteria_actual.vector_cotas) * np.copy(
                self.instancia_estanteria_actual.vector_centro_celda))

        self.vector_posicion_actual = vector_posicion_final

        # Retorna el tiempo total de ejecución
        return self._registrarMovimiento(vector_posicion_inicial, vector_posicion_final,
                                         self._tiempo(vector_posicion_inicial, vector_posicion_final),
                                         False, False)

    def cargarItemDeEstanteria(self, vector_coordenada_recuperacion):
        # La grúa recoge un ítem de una posición dada en la estantería
        # El ítem se añade a la carga de la grúa y se elimina de la estantería
        instancia_item = self.instancia_estanteria_actual.matriz_contenedor_items[
            vector_coordenada_recuperacion]
        self.vector_carga.append(instancia_item)
        self.instancia_estanteria_actual.matriz_contenedor_items[vector_coordenada_recuperacion] = 0

        return self._registrarCarga('P', *[int(escalar_indice) for escalar_indice in vector_coordenada_recuperacion])

    def soltarItemEnEstanteria(self, vector_coordenada_almacenamiento):
        # La grúa suelta un ítem en una posición dada en la estantería
        # El ítem se elimina de la carga de la grúa y se añade a la estantería
        instancia_item = self.vector_carga.pop()
        self.instancia_estanteria_actual.matriz_contenedor_items[
            vector_coordenada_almacenamiento] = instancia_item

        return self._registrarCarga('D', *[int(escalar_indice) for escalar_indice in vector_coordenada_almacenamiento])

    # Métodos para interactuar con el punto E/S

    def irAPuntoES(self, instancia_PuntoES):
        # La grúa se mueve a un punto E/S dado
        # La posición actual de la grúa se actualiza a la posición del punto E/S

        vector_posicion_inicial = np.copy(self.vector_posicion_actual)
        booleano_en_es_inicial = self._enES()
        self.instancia_estanteria_actual = 0
        self.instancia_puntoES_actual = instancia_PuntoES
        vector_posicion_final = np.array(self.instancia_puntoES_actual.vector_posicion, dtype=float)
        self.vector_posicion_actual = np.copy(vector_posicion_final)

        # Retorna el tiempo total de ejecución
        return self._registrarMovimiento(vector_posicion_inicial, vector_posicion_final,
                                         self._tiempo(vector_posicion_inicial, vector_posicion_final),
                                         booleano_en_es_inicial, True)

    def cargarItemPuntoES(self, indice=-1):
        # La grúa recoge el ítem del punto E/S en la posición dada por 'indice'
        # Si 'indice' no se proporciona, se recoge el último ítem (comportamiento de pop())
        # El ítem se elimina del punto E/S y se carga en la grúa
        instancia_item = self.instancia_puntoES_actual.matriz_contenedor_temporal_items.pop(
            indice)
        self.vector_carga.append(instancia_item)

        return self._registrarCarga('J', indice)

    def soltarItemEnPuntoES(self):
        # La grúa suelta un ítem en el punto E/S
        # El ítem se añade al final del contenedor temporal del punto E/S
        instancia_item = self.vector_carga.pop()
        self.instancia_puntoES_actual.matriz_contenedor_temporal_items.append(
            instancia_item)

        return self._registrarCarga('I')

    def soltarItemEnPuntoESSalida(self):
        # La grúa suelta un ítem en el punto E/S
        # El ítem se añade al final del contenedor de salida del punto E/S
        instancia_item = self.vector_carga.pop()
        self.instancia_puntoES_actual.matriz_contenedor_salida_items.append(
            instancia_item)

        return self._registrarCarga('S')

    # Métodos para reposicionar la grúa

    def volverAplano(self):
        # La grúa se mueve a la altura del riel (posición z de reposo)
        vector_posicion_inicial, vector_posicion_final, escalar_tiempo = self._aplanar()

        # Retorna el tiempo total de ejecución
        return self._registrarMovimiento(vector_posicion_inicial, vector_posicion_final, escalar_tiempo, self._enES(), self._enES())

    def volverAReposo(self):
        # La grúa vuelve a su posición de reposo
        # La grúa ya no está en ninguna estantería o punto E/S
        vector_posicion_inicial = np.copy(self.vector_posicion_actual)
        vector_posicion_final = np.copy(self.vector_posicion_resposo)
        booleano_en_es_inicial = self._enES()

        self.instancia_estanteria_actual = 0
        self.instancia_puntoES_actual = 0
        self.vector_posicion_actual = np.copy(vector_posicion_final)

        # Retorna el tiempo total de ejecución
        return self._registrarMovimiento(vector_posicion_inicial, vector_posicion_final,
                                         self._tiempo(vector_posicion_inicial, vector_posicion_final),
                                         booleano_en_es_inicial, True)

    def __repr__(self) -> str:
        # Este método devuelve una representación en cadena de caracteres del objeto
        # Incluye la posición actual de la grúa y su velocidad
        return f'{self.vector_posicion_actual}|{self.vector_velocidad}'
