"""Registro de políticas de asignación disponibles."""
from asrs.politicas.aleatoria import PoliticaAleatoria
from asrs.politicas.basada_en_clases import PoliticaBasadaEnClases, PoliticaBasadaEnClasesReplanificada
from asrs.politicas.full_turnover import PoliticaFullTurnover, PoliticaFullTurnoverReplanificada
from asrs.politicas.mas_cercana_es import PoliticaMasCercanaAlES
from asrs.politicas.mas_cercana_grua import PoliticaMasCercanaALaGrua
from asrs.politicas.recuperacion import PoliticaRecuperacionFCFS, PoliticaRecuperacionFCFSAccesible

POLITICAS_ASIGNACION = {
    'aleatoria': PoliticaAleatoria,
    'basada_en_clases': PoliticaBasadaEnClases,
    'basada_en_clases_replanificada': PoliticaBasadaEnClasesReplanificada,
    'full_turnover': PoliticaFullTurnover,
    'full_turnover_replanificada': PoliticaFullTurnoverReplanificada,
    'mas_cercana_es': PoliticaMasCercanaAlES,
    'mas_cercana_grua': PoliticaMasCercanaALaGrua,
}


def crear_politica(txt_clave, instancia_estanteria, instancia_grua, instancia_puntoES, config):
    # Construye la política de asignación a partir de su clave de configuración
    if txt_clave not in POLITICAS_ASIGNACION:
        raise ValueError(f"Política desconocida '{txt_clave}'. Disponibles: {', '.join(POLITICAS_ASIGNACION)}")
    return POLITICAS_ASIGNACION[txt_clave](instancia_estanteria, instancia_grua, instancia_puntoES, config)


POLITICAS_RECUPERACION = {
    'fcfs': PoliticaRecuperacionFCFS,
    'fcfs_accesible': PoliticaRecuperacionFCFSAccesible,
}


def crear_politica_recuperacion(txt_clave):
    # Construye la política de recuperación a partir de su clave de configuración
    if txt_clave not in POLITICAS_RECUPERACION:
        raise ValueError(f"Política de recuperación desconocida '{txt_clave}'. Disponibles: {', '.join(POLITICAS_RECUPERACION)}")
    return POLITICAS_RECUPERACION[txt_clave]()
