"""Gestiona el entrenamiento de agentes de RL como procesos aparte y guarda lo que emiten, para que la página lo consulte."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from asrs.config import config_a_json, config_desde_json

LIBRERIAS = ("torch", "gymnasium", "stable_baselines3", "sb3_contrib")
LIMITE_PASOS = 2_000_000
LIMITE_FOTOS = 120                    # fotos de la red que se conservan, si hay más se descarta una de cada dos


def faltan_librerias():
    return [nombre for nombre in LIBRERIAS if importlib.util.find_spec(nombre) is None]


class Entrenamiento:
    ESTADOS_FINALES = ("listo", "error", "cancelado")

    def __init__(self, dic_config, txt_algoritmo, escalar_pasos, escalar_entornos, escalar_estados, escalar_pruebas, escalar_semilla):
        self.id = uuid.uuid4().hex[:12]
        self.creado = time.time()
        self.estado = "iniciando"
        self.algoritmo = txt_algoritmo
        self.inicio, self.fin, self.error = None, None, None
        self.referencias, self.evaluacion = {}, {}
        self.metricas, self.fotos = [], []
        self.secuencia = 0
        self.cancelado = False
        self.cerrojo = threading.Lock()
        self.cola_errores = []

        # La configuración se pasa por un archivo temporal, el proceso la valida de nuevo con config_desde_json
        self.archivo_config = os.path.join(tempfile.gettempdir(), f"asrs_config_{self.id}.json")
        with open(self.archivo_config, "w", encoding="utf-8") as archivo:
            json.dump(config_a_json(dic_config), archivo)
        vector_comando = [sys.executable, "-W", "ignore", "-m", "asrs.rl.entrenamiento_vivo", self.archivo_config, "--algoritmo", txt_algoritmo,
                          "--pasos", str(escalar_pasos), "--entornos", str(escalar_entornos), "--estados", str(escalar_estados),
                          "--pruebas", str(escalar_pruebas), "--semilla", str(escalar_semilla)]
        txt_raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.proceso = subprocess.Popen(vector_comando, cwd=txt_raiz, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                        errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        threading.Thread(target=self._leer_salida, daemon=True).start()
        threading.Thread(target=self._leer_errores, daemon=True).start()

    def _leer_errores(self):
        for txt_linea in self.proceso.stderr:
            self.cola_errores = (self.cola_errores + [txt_linea.rstrip()])[-15:]

    def _leer_salida(self):
        for txt_linea in self.proceso.stdout:
            txt_linea = txt_linea.strip()
            if not txt_linea.startswith("{"):
                continue
            try:
                dic_evento = json.loads(txt_linea)
            except ValueError:
                continue
            with self.cerrojo:
                self._aplicar(dic_evento)
        self.proceso.wait()
        with self.cerrojo:
            if self.estado not in self.ESTADOS_FINALES:
                if self.cancelado:
                    self.estado = "cancelado"
                else:
                    self.estado = "error"
                    self.error = self.error or ("El proceso de entrenamiento terminó de forma inesperada.\n" + "\n".join(self.cola_errores))
        try:
            os.remove(self.archivo_config)
        except OSError:
            pass

    def _aplicar(self, dic_evento):
        txt_tipo = dic_evento.get("tipo")
        if txt_tipo == "inicio":
            self.inicio = dic_evento
            self.estado = "referencias"
        elif txt_tipo == "referencia":
            self.referencias[dic_evento["clave"]] = dic_evento["media"]
        elif txt_tipo == "metrica":
            self.secuencia += 1
            dic_evento["n"] = self.secuencia
            self.metricas.append(dic_evento)
            if self.estado in ("iniciando", "referencias"):
                self.estado = "entrenando"
        elif txt_tipo == "snapshot":
            self.secuencia += 1
            dic_evento["n"] = self.secuencia
            self.fotos.append(dic_evento)
            if len(self.fotos) > LIMITE_FOTOS:
                self.fotos = self.fotos[::2][:-1] + self.fotos[-1:]
            if self.estado in ("iniciando", "referencias"):
                self.estado = "entrenando"
        elif txt_tipo == "evaluacion":
            self.evaluacion[dic_evento["clave"]] = dic_evento["media"]
            self.estado = "evaluando"
        elif txt_tipo == "fin":
            self.fin = dic_evento
            self.estado = "listo"
        elif txt_tipo == "error":
            self.error = dic_evento.get("mensaje", "Error desconocido.")
            self.estado = "error"

    def cancelar(self):
        self.cancelado = True
        if self.proceso.poll() is None:
            # En Windows se termina todo el árbol: el proceso principal y los entornos paralelos
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.proceso.pid), "/T", "/F"], capture_output=True)
            else:
                self.proceso.kill()

    def publico(self, escalar_desde):
        # Todo lo ocurrido después del número de secuencia indicado, así la página pide solo lo nuevo
        with self.cerrojo:
            return {"id": self.id, "estado": self.estado, "algoritmo": self.algoritmo, "inicio": self.inicio, "referencias": self.referencias,
                    "evaluacion": self.evaluacion, "fin": self.fin, "error": self.error, "secuencia": self.secuencia,
                    "metricas": [m for m in self.metricas if m["n"] > escalar_desde], "fotos": [f for f in self.fotos if f["n"] > escalar_desde]}


class GestorEntrenamiento:
    def __init__(self):
        self.entrenamientos = {}
        self.cerrojo = threading.Lock()

    def activo(self):
        with self.cerrojo:
            return next((e for e in self.entrenamientos.values() if e.estado not in Entrenamiento.ESTADOS_FINALES), None)

    def iniciar(self, dic_solicitud):
        # Valida la solicitud y lanza el proceso. ValueError si algo no es válido, RuntimeError si ya hay un entrenamiento en curso
        vector_faltan = faltan_librerias()
        if vector_faltan:
            raise ValueError("Faltan librerías para entrenar: " + ", ".join(vector_faltan) + ". Instálalas con: pip install -r requirements-rl.txt")
        txt_algoritmo = dic_solicitud.get("algoritmo")
        if txt_algoritmo not in ("dqn", "ppo"):
            raise ValueError("Algoritmo: debe ser 'dqn' o 'ppo'.")
        escalar_pasos = _entero(dic_solicitud.get("pasos"), "Pasos de entrenamiento", 1000, LIMITE_PASOS)
        escalar_entornos = _entero(dic_solicitud.get("entornos"), "Entornos en paralelo", 1, max(1, (os.cpu_count() or 2)))
        escalar_estados = _entero(dic_solicitud.get("estados", 16), "Estados de entrenamiento", 1, 200)
        escalar_pruebas = _entero(dic_solicitud.get("pruebas", 4), "Estados de prueba", 1, 50)
        escalar_semilla = _entero(dic_solicitud.get("semilla", 0), "Semilla", 0, 1_000_000)
        dic_config = config_desde_json(dic_solicitud.get("config", {}))
        with self.cerrojo:
            if any(e.estado not in Entrenamiento.ESTADOS_FINALES for e in self.entrenamientos.values()):
                raise RuntimeError("Ya hay un entrenamiento en curso. Espera a que termine o cancélalo.")
            entrenamiento = Entrenamiento(dic_config, txt_algoritmo, escalar_pasos, escalar_entornos, escalar_estados, escalar_pruebas, escalar_semilla)
            self.entrenamientos[entrenamiento.id] = entrenamiento
            for antiguo in [e for e in sorted(self.entrenamientos.values(), key=lambda e: e.creado) if e.estado in Entrenamiento.ESTADOS_FINALES][:-3]:
                del self.entrenamientos[antiguo.id]
        return entrenamiento

    def buscar(self, txt_id):
        return self.entrenamientos.get(txt_id)

    def cancelar_todo(self):
        for entrenamiento in list(self.entrenamientos.values()):
            entrenamiento.cancelar()


def _entero(valor, txt_nombre, escalar_minimo, escalar_maximo):
    try:
        escalar = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"{txt_nombre}: falta el valor o no es un número.")
    if escalar != int(escalar) or not escalar_minimo <= escalar <= escalar_maximo:
        raise ValueError(f"{txt_nombre}: debe ser un entero entre {escalar_minimo} y {escalar_maximo}.")
    return int(escalar)
