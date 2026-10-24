"""Controlador que ejecuta la secuencia de operaciones con una política de asignación y una de recuperación."""
import numpy as np

from asrs.dominio.item import Item
from asrs.metricas import Metricas
from asrs.politicas.recuperacion import PoliticaRecuperacionFCFS


class ControladorGrua:
    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, vector_secuencia_operaciones,
                 instancia_politica_asignacion, instancia_politica_recuperacion = None) -> None:

        # Se inicializan los atributos de clase
        self.instancia_grua = instancia_grua
        self.instancia_estanteria = instancia_estanteria
        self.vector_secuencia_operaciones = vector_secuencia_operaciones
        self.instancia_puntoES = instancia_puntoES
        self.instancia_metricas = Metricas()

        # La política de asignación se usa para almacenar y también para reubicar: reubicar es almacenar desde otro punto de referencia
        self.instancia_politica_asignacion = instancia_politica_asignacion
        self.instancia_politica_recuperacion = instancia_politica_recuperacion if instancia_politica_recuperacion is not None else PoliticaRecuperacionFCFS()

    def listarObstrucciones(self, vector_indice_celda_objetivo):
        # Esta función lista los indices de las celdas que obstruyen a la celda objetivo

        # Se recupera la matriz que guarda los datos sobre los ítems almacenados
        matriz_contenedor_items = self.instancia_estanteria.matriz_contenedor_items
        # Se consulta la cantidad de capas o niveles de profundidad que tiene la estantería
        escalar_cantidad_capas = matriz_contenedor_items.shape[2]

        # Se desempaquetan los indices de la celda objetivo en sus componentes
        escalar_indice_fila_obj, escalar_indice_columna_obj, escalar_indice_capa_obj = vector_indice_celda_objetivo

        # Se crea una lista que sevirá para almacenar los índices de las celdas que obstruyen
        vector_lista_indices_obstruyen = []

        # Se recorre desde la capa más superficial, bajando hasta llegar justamente antes de la profundidad objetivo
        # En el ciclo se evalúa si las cedas estan ocupadas por algún ítem
        # Sí se encuentran ítems en el recorrido de descenso entonces los indices se van agregando a la lista de indices que obstruyen
        for escalar_indice_capa in reversed(range(escalar_indice_capa_obj + 1, escalar_cantidad_capas)):
            if isinstance(matriz_contenedor_items[escalar_indice_fila_obj, escalar_indice_columna_obj, escalar_indice_capa], Item):
                vector_lista_indices_obstruyen.append((escalar_indice_fila_obj, escalar_indice_columna_obj, escalar_indice_capa))

        return vector_lista_indices_obstruyen

    def devolverItemsDesdePuntoES(self):
        # Retorna a la estantería los ítems que reposan temporalmente en el puntoES, el último en llegar es el primero en volver
        instancia_grua = self.instancia_grua
        instancia_puntoES = self.instancia_puntoES
        instancia_estanteria = self.instancia_estanteria
        escalar_tiempo = 0

        while len(instancia_puntoES.matriz_contenedor_temporal_items):
            instancia_item_retornar = instancia_puntoES.matriz_contenedor_temporal_items[-1]
            escalar_sku = instancia_item_retornar.escalar_sku

            vector_indice_retorno = self.instancia_politica_asignacion.asignar(escalar_sku, vector_posicion_referencia = instancia_puntoES.vector_posicion)
            if not len(vector_indice_retorno):
                # Si la zona del SKU no tiene espacio accesible se relaja la zonificación
                vector_indice_retorno = self.instancia_politica_asignacion.asignar(escalar_sku, booleano_restringir_zona = False, vector_posicion_referencia = instancia_puntoES.vector_posicion)
            if not len(vector_indice_retorno):
                # No hay ninguna celda accesible, el ítem permanece en el puntoES
                break

            escalar_tiempo += instancia_grua.irAPuntoES(instancia_puntoES)
            escalar_tiempo += instancia_grua.cargarItemPuntoES()
            escalar_tiempo += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
            escalar_tiempo += instancia_grua.moverEnEstanteria(vector_indice_retorno)
            escalar_tiempo += instancia_grua.soltarItemEnEstanteria(vector_indice_retorno)
            escalar_tiempo += instancia_grua.volverAplano()
            escalar_tiempo += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

        return escalar_tiempo

    def reubicar(self, vector_lista_indices_obstruyen):
        # Esta función realiza las actividades de reubicación de los ítems que obstruyen según la lista recbida como argumento

        # Se llaman a las variables relevantes
        instancia_grua = self.instancia_grua
        instancia_puntoES = self.instancia_puntoES
        instancia_estanteria = self.instancia_estanteria

        # Declaración de banderas
        booleano_irAEstanteria = False
        booleano_reubicacion_predeterminada = False

        # Se inicializan acumuladores de tiempo
        escalar_tiempo_total_reubicaciones_regulares = 0
        escalar_tiempo_total_reubicaciones_predeterminadas = 0

        # Se construye una matriz de restricciones que señaliza como no factible la reubicación dentro del mismo canal de la celda objetivo
        # Esto evita que el ítem retirado vuelva a obstruir la celda objetivo
        matriz_no_estatico = np.ones(instancia_estanteria.matriz_contenedor_items.shape, dtype=bool)
        matriz_no_estatico[vector_lista_indices_obstruyen[0][0], vector_lista_indices_obstruyen[0][1], :] = False

        # Por cada celda obstruida se realiza la reubicación respectiva
        for vector_indice_obstruye in vector_lista_indices_obstruyen:

            # Se consulta el ítem que obstruye y de qué SKU es
            instancia_item_bloquea = instancia_estanteria.matriz_contenedor_items[vector_indice_obstruye]
            escalar_sku = instancia_item_bloquea.escalar_sku

            # Se busca la mejor celda libre, accesible, de la zona del SKU y fuera del canal objetivo
            vector_indice_reubicacion = self.instancia_politica_asignacion.asignar(escalar_sku, matriz_extra = matriz_no_estatico,
                                                                       vector_posicion_referencia = instancia_estanteria.posicionesCeldas()[vector_indice_obstruye])

            # Se evalúa si hay si hay alguna ubicación factible para reubicar
            if len(vector_indice_reubicacion):

                # en caso verdaro se evalua sí la grúa ya está en la estantería
                if not booleano_irAEstanteria:
                    # Si aún no está en la estantería entonces va a la estantería
                    escalar_tiempo_total_reubicaciones_regulares += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                    booleano_irAEstanteria = True

                # Se ejecuta la rutina de reubicación regular(reubicación dentro de estantería)
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.moverEnEstanteria(vector_indice_obstruye)
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.cargarItemDeEstanteria(vector_indice_obstruye)
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.volverAplano()
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.moverEnEstanteria(vector_indice_reubicacion)
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.soltarItemEnEstanteria(vector_indice_reubicacion)
                escalar_tiempo_total_reubicaciones_regulares += instancia_grua.volverAplano()

            else:  # Ejecutar reubicacion predeterminada

                # Se activa la bandera indicando que se realizo reubicación predeterminada
                booleano_reubicacion_predeterminada = True

                # se evalua sí la grúa ya está en la estantería
                if not booleano_irAEstanteria:
                    # Si aún no está en la estantería entonces va a la estantería
                    escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                    booleano_irAEstanteria = True

                # Se ejecuta la rutina de reubicación predeterminada(reubicación en el puntoES)
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.moverEnEstanteria(vector_indice_obstruye)
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.cargarItemDeEstanteria(vector_indice_obstruye)
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.volverAplano()
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.irAPuntoES(instancia_puntoES)
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.soltarItemEnPuntoES()
                escalar_tiempo_total_reubicaciones_predeterminadas += instancia_grua.volverAplano()

                # La grúa quedo fuera de la estantería entonces se desactiva la bandera
                booleano_irAEstanteria = False

        # Finalizado el proceso de reubicación la grúa queda a la altura del riel donde soltó el último ítem
        # No vuelve a la esquina: desde ahí va directo a la celda objetivo, y ese tramo cuenta en la recuperación

        # Se registran las métricas
        self.instancia_metricas.registrarReubicacion(escalar_tiempo_total_reubicaciones_regulares + escalar_tiempo_total_reubicaciones_predeterminadas)

        return booleano_reubicacion_predeterminada, escalar_tiempo_total_reubicaciones_regulares, escalar_tiempo_total_reubicaciones_predeterminadas

    def almacenar(self, vector_indice_almacenamiento):
        # Esta función realiza las actividades de almacenamiento, se almacena el ítem en el índice que toma como argumento

        # Se llaman a las variables relevantes
        instancia_grua = self.instancia_grua
        instancia_estanteria = self.instancia_estanteria
        instancia_puntoES = self.instancia_puntoES

        # Se declara una bandera que representa el exito de tarea de almacenamiento
        booleano_item_almacenado = False

        # Se inicializan acumuladores de tiempo
        escalar_tiempo_almacenamiento = 0
        escalar_tiempo_total_reubicaciones_regulares_almacenamiento = 0
        escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento = 0

        # Se consultan los indices que obstruyen
        vector_lista_indices_obstruyen = self.listarObstrucciones(vector_indice_almacenamiento)

        # Se evalúa si hay celdas que obstruyen
        if len(vector_lista_indices_obstruyen):
            # En caso de verdadero se realiza la reubicación con el método 'reubicar'
            booleano_reubicacion_predeterminada, escalar_tiempo_total_reubicaciones_regulares_almacenamiento, escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento = self.reubicar(vector_lista_indices_obstruyen)

            # Se evalúa si se realizó reubicación predeterminada
            if booleano_reubicacion_predeterminada:

                # En este punto el ítem ya se encuentra despejado, entonces se realiza la rutina de almacenamiento
                escalar_tiempo_almacenamiento += instancia_grua.irAPuntoES(instancia_puntoES)
                escalar_tiempo_almacenamiento += instancia_grua.cargarItemPuntoES(indice=0)
                escalar_tiempo_almacenamiento += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                escalar_tiempo_almacenamiento += instancia_grua.moverEnEstanteria(vector_indice_almacenamiento)
                escalar_tiempo_almacenamiento += instancia_grua.soltarItemEnEstanteria(vector_indice_almacenamiento)
                escalar_tiempo_almacenamiento += instancia_grua.volverAplano()
                escalar_tiempo_almacenamiento += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

                # Se activa la bandera por el exito de la tarea
                booleano_item_almacenado = True

                # Luego, los ítems que reponsan temporalmente en el puntoES deben ser retornados a la estantería
                escalar_tiempo_retorno = self.devolverItemsDesdePuntoES()
                escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento += escalar_tiempo_retorno

                # El tiempo de retorno también hace parte del tiempo total de reubicación
                self.instancia_metricas.sumarTiempoReubicacion(escalar_tiempo_retorno)

        # Luego de realizar todas las reubicaciones si las hubiere, se evalúa si el ítem objetivo aún no ha sido almacenado
        if not booleano_item_almacenado:
            # en caso de que aún no se haya almacenado, entonces se realiza la rutina de almacenamiento
            escalar_tiempo_almacenamiento += instancia_grua.irAPuntoES(instancia_puntoES)
            escalar_tiempo_almacenamiento += instancia_grua.cargarItemPuntoES()
            escalar_tiempo_almacenamiento += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
            escalar_tiempo_almacenamiento += instancia_grua.moverEnEstanteria(tuple(vector_indice_almacenamiento))
            escalar_tiempo_almacenamiento += instancia_grua.soltarItemEnEstanteria(tuple(vector_indice_almacenamiento))
            escalar_tiempo_almacenamiento += instancia_grua.volverAplano()
            escalar_tiempo_almacenamiento += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

        # Finalizado el la tarea de almacenamiento, se procede a dejar la grúa en su posición de reposo
        escalar_tiempo_almacenamiento += instancia_grua.volverAReposo()

        # Se registran las métricas en el caso de que los eventos relacionadas se hayan ejecutado
        self.instancia_metricas.registrarAlmacenamiento(escalar_tiempo_almacenamiento)
        if escalar_tiempo_total_reubicaciones_regulares_almacenamiento:
            self.instancia_metricas.registrarReubicacionRegularAlmacenamiento(escalar_tiempo_total_reubicaciones_regulares_almacenamiento)
        if escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento:
            self.instancia_metricas.registrarReubicacionPredeterminadaAlmacenamiento(escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento)
        if escalar_tiempo_total_reubicaciones_regulares_almacenamiento or escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento:
            self.instancia_metricas.registrarReubicacionAlmacenamiento(escalar_tiempo_total_reubicaciones_regulares_almacenamiento + escalar_tiempo_total_reubicaciones_predeterminadas_almacenamiento)

    def recuperar(self, vector_indice_recuperacion):
        # Esta función se encarga de procesar y mandar la tarea de recuperación

        # Se llaman a las variables relevantes
        instancia_grua = self.instancia_grua
        instancia_estanteria = self.instancia_estanteria
        instancia_puntoES = self.instancia_puntoES

        # Se inicializa una bandera para llevar el control del exito de tarea
        booleano_item_recuperado = False

        # Se incializan acumuladores de tiempo
        escalar_tiempo_recuperacion = 0
        escalar_tiempo_total_reubicaciones_regulares_recuperacion = 0
        escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion = 0

        # Se listan obstrupciones sobre la celda objetivo
        vector_lista_indices_obstruyen = self.listarObstrucciones(vector_indice_recuperacion)

        # Se evalúa si hay celdas que obstruyen
        if len(vector_lista_indices_obstruyen):
            # En caso de verdadero se realiza la reubicación con el método 'reubicar'
            booleano_reubicacion_predeterminada, escalar_tiempo_total_reubicaciones_regulares_recuperacion, escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion = self.reubicar(vector_lista_indices_obstruyen)

            # Se evalúa si se realizó reubicación predeterminada
            if booleano_reubicacion_predeterminada:

                # En este punto el ítem ya se encuentra despejado, entonces se realiza la rutina de recuperación
                escalar_tiempo_recuperacion += instancia_grua.entrarEnEstanteria(instancia_estanteria)
                escalar_tiempo_recuperacion += instancia_grua.moverEnEstanteria(vector_indice_recuperacion)
                escalar_tiempo_recuperacion += instancia_grua.cargarItemDeEstanteria(vector_indice_recuperacion)
                escalar_tiempo_recuperacion += instancia_grua.volverAplano()
                escalar_tiempo_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
                escalar_tiempo_recuperacion += instancia_grua.irAPuntoES(instancia_puntoES)
                escalar_tiempo_recuperacion += instancia_grua.soltarItemEnPuntoESSalida()
                escalar_tiempo_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

                # Se activa la bandera por el exito de la tarea
                booleano_item_recuperado = True

                # Luego, los ítems que reponsan temporalmente en el puntoES deben ser retornados a la estantería
                escalar_tiempo_retorno = self.devolverItemsDesdePuntoES()
                escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion += escalar_tiempo_retorno

                # El tiempo de retorno también hace parte del tiempo total de reubicación
                self.instancia_metricas.sumarTiempoReubicacion(escalar_tiempo_retorno)

        # Luego de realizar todas las reubicaciones si las hubiere, se evalúa si el ítem objetivo aún no ha sido recuperado
        if not booleano_item_recuperado:
            # en caso de que aún no se haya almacenado, entonces se realiza la rutina de recuperación
            # Si hubo reubicaciones regulares la grúa sigue en la estantería y va directo a la celda objetivo
            escalar_tiempo_recuperacion += instancia_grua.entrarEnEstanteria(instancia_estanteria)
            escalar_tiempo_recuperacion += instancia_grua.moverEnEstanteria(vector_indice_recuperacion)
            escalar_tiempo_recuperacion += instancia_grua.cargarItemDeEstanteria(vector_indice_recuperacion)
            escalar_tiempo_recuperacion += instancia_grua.volverAplano()
            escalar_tiempo_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)
            escalar_tiempo_recuperacion += instancia_grua.irAPuntoES(instancia_puntoES)
            escalar_tiempo_recuperacion += instancia_grua.soltarItemEnPuntoESSalida()
            escalar_tiempo_recuperacion += instancia_grua.irAEsquinaEstanteria(instancia_estanteria)

        # Finalizda la actividad de recuperación se da la orden a la grúa de que regrese a reposo
        escalar_tiempo_recuperacion += instancia_grua.volverAReposo()

        # Se registran las métricas en el caso de que los eventos relacionadas se hayan ejecutado
        self.instancia_metricas.registrarRecuperacion(escalar_tiempo_recuperacion)
        if escalar_tiempo_total_reubicaciones_regulares_recuperacion:
            self.instancia_metricas.registrarReubicacionRegularRecuperacion(escalar_tiempo_total_reubicaciones_regulares_recuperacion)
        if escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion:
            self.instancia_metricas.registrarReubicacionPredeterminadaRecuperacion(escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion)
        if escalar_tiempo_total_reubicaciones_regulares_recuperacion or escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion:
            self.instancia_metricas.registrarReubicacionRecuperacion(escalar_tiempo_total_reubicaciones_regulares_recuperacion + escalar_tiempo_total_reubicaciones_predeterminadas_recuperacion)

    def ejecutarSecuencia(self):
        # Esta funcion se encarga de ejecutar administrativamente la secuencia de operaciones

        # Se llaman a las variables relevantes
        vector_secuencia_operaciones = self.vector_secuencia_operaciones

        # Se establece que el almacenamiento es 0 y la recuperación es 1, esto para facilitar la lectura del código
        escalar_almacenamiento = 0
        escalar_recuperacion = 1

        # Se recorre la secuencia de operaciones, operación por operación
        for vector_operacion in vector_secuencia_operaciones:

            # Se desmpaqueta la linea de operación, que se compone de un identificador de actividad, y un identificador del SKU
            escalar_actividad, escalar_sku = vector_operacion

            # se evalúa si la actvidad es almacenar
            if escalar_actividad == escalar_almacenamiento:

                # En caso verdadero se consulta la posición facitble en donde almacenar
                # La política solo elige celdas accesibles, 'almacenar' conserva el manejo de obstrucciones por si se usa otra política
                vector_indice_almacenamiento = self.instancia_politica_asignacion.asignar(escalar_sku)

                # Se evalúa si dicha posicion factible existe
                if len(vector_indice_almacenamiento):
                    # En caso verdadero, entonces realizar el almacenamiento del ítem en la posición establecida
                    self.instancia_grua.registrarEvento('T', escalar_actividad, escalar_sku, 1)
                    instancia_item_nuevo = Item(escalar_sku)
                    self.instancia_puntoES.matriz_contenedor_temporal_items.append(instancia_item_nuevo)
                    self.instancia_grua.registrarEvento('A', escalar_sku)
                    self.almacenar(vector_indice_almacenamiento)
                else:
                    self.instancia_grua.registrarEvento('T', escalar_actividad, escalar_sku, 0)
                    # En caso de que no exista posición factible, de forma predeterminada se descarta la operación de almacenamiento
                    # Este caso representa que no hay espacio disponible para almacenar el ítem
                    pass

            # O se evalúa si es recuperación
            elif escalar_actividad == escalar_recuperacion:

                # En caso verdadero se consulta si hay algun ítem disponible que coincida con el SKU solicitado
                vector_indice_recuperacion = self.instancia_politica_recuperacion.seleccionar(self.instancia_estanteria, escalar_sku)

                # Se evalúa si efectivamente se encontro alguna celda que contenga el ítem solicitado
                if len(vector_indice_recuperacion):

                    # En caso verdadero, se realiza la recuperación del ítem
                    self.instancia_grua.registrarEvento('T', escalar_actividad, escalar_sku, 1)
                    self.recuperar(vector_indice_recuperacion)

                else:
                    self.instancia_grua.registrarEvento('T', escalar_actividad, escalar_sku, 0)
                    # En caso de no exista el ítem,  entonces se omite la operación de recuperación
                    pass
