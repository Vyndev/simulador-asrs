"""Registro de políticas de asignación disponibles."""
from asrs.politicas.aleatoria import PoliticaAleatoria
from asrs.politicas.basada_en_clases import PoliticaBasadaEnClases
from asrs.politicas.full_turnover import PoliticaFullTurnover
from asrs.politicas.mas_cercana_es import PoliticaMasCercanaAlES
from asrs.politicas.mas_cercana_grua import PoliticaMasCercanaALaGrua

POLITICAS_ASIGNACION = {
    'aleatoria': PoliticaAleatoria,
    'basada_en_clases': PoliticaBasadaEnClases,
    'full_turnover': PoliticaFullTurnover,
    'mas_cercana_es': PoliticaMasCercanaAlES,
    'mas_cercana_grua': PoliticaMasCercanaALaGrua,
}


def crear_politica(txt_clave, instancia_estanteria, instancia_grua, instancia_puntoES, config):
    # Construye la política de asignación a partir de su clave de configuración
    if txt_clave not in POLITICAS_ASIGNACION:
        raise ValueError(f"Política desconocida '{txt_clave}'. Disponibles: {', '.join(POLITICAS_ASIGNACION)}")
    return POLITICAS_ASIGNACION[txt_clave](instancia_estanteria, instancia_grua, instancia_puntoES, config)
