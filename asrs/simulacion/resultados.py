"""Escritura de resultados en CSV."""
import csv


def guardar_resultados_csv(resultados, ruta = 'resultados_simulacion.csv'):

    # Es una función de utilidad, ayuda a escribir los resultados de la simulación en un archivo CSV
    with open(ruta, 'w', newline='') as archivo:

        # Dependiendo de la configuración regional del lector de hojas de cálculo en ciertas ocaciones el .csv no se visualiza correctamente
        # Entonces, se recomienda modificar el delimitador/delimiter según sea necesario
        writer = csv.writer(archivo, delimiter=';')
        writer.writerow(["Politica_Asignacion",
                              "Cant_Almacenamientos", "TiempoTotal_Almacenamiento", "DesvEst_Tiempo_Almacenamiento", #A
                              "Cant_Recuperaciones", "TiempoTotal_Recuperacion", "DesvEst_Tiempo_Recuperacion", #B
                              "Cant_Reubicaciones", "TiempoTotal_Reubicacion", "DesvEst_Tiempo_Reubicacion", #C
                              "Cant_ReubicacionesAlmacenamiento", "TiempoTotal_ReubicacionesAlmacenamiento", "DesvEst_Tiempo_ReubicacionesAlmacenamiento", #D
                              "Cant_ReubicacionesRecuperacion", "TiempoTotal_ReubicacionesRecuperacion", "DesvEst_Tiempo_ReubicacionesRecuperacion", #E
                              "Cant_ReubicacionesRegularesAlmacenamiento", "TiempoTotal_ReubicacionesRegularesAlmacenamiento", "DesvEst_Tiempo_ReubicacionesRegularesAlmacenamiento", #F
                              "Cant_ReubicacionesPredeterminadasAlmacenamiento", "TiempoTotal_ReubicacionesPredeterminadasAlmacenamiento", "DesvEst_Tiempo_ReubicacionesPredeterminadasAlmacenamiento", #G
                              "Cant_ReubicacionesRegularesRecuperacion", "TiempoTotal_ReubicacionesRegularesRecuperacion", "DesvEst_Tiempo_ReubicacionesRegularesRecuperacion", #H
                              "Cant_ReubicacionesPredeterminadasRecuperacion", "TiempoTotal_ReubicacionesPredeterminadasRecuperacion", "DesvEst_Tiempo_ReubicacionesPredeterminadasRecuperacion", #I
                              "Cant_Operaciones", "TiempoTotal_Operacion"])
        writer.writerows(resultados)
