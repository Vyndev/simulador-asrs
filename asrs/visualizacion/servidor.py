"""Servidor local de la herramienta: sirve la página y ejecuta la simulación con los parámetros que recibe.

Solo escucha en 127.0.0.1. Las simulaciones largas corren en segundo plano como "trabajos" y la página consulta su progreso.
"""
import json
import multiprocessing as mp
import socket
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from asrs.config import LIMITE_SKUS, LIMITE_TAREAS, cargar_configuracion, config_a_json, config_desde_json
from asrs.generacion.datos_reales import leer_tareas, resumir_tareas
from asrs.simulacion.estadistica import resumir
from asrs.simulacion.simulacion import Simulacion
from asrs.visualizacion.generador import generar_datos, leer_plantilla

LIMITE_CUERPO = 1_000_000          # bytes aceptados en una solicitud
TRABAJOS_GUARDADOS = 20            # trabajos terminados que se conservan en memoria


class Cancelado(Exception):
    pass


class Trabajo:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.estado = 'en_cola'      # en_cola, ejecutando, listo, error, cancelado
        self.mensaje = ''
        self.progreso = 0
        self.total = 1
        self.resultado = None
        self.error = None
        self.cancelado = False
        self.creado = time.time()
        self._cerrojo = threading.Lock()

    def avanzar(self):
        with self._cerrojo:
            self.progreso += 1
        if self.cancelado:
            raise Cancelado()

    def estado_publico(self):
        return {"id": self.id, "estado": self.estado, "mensaje": self.mensaje, "progreso": self.progreso, "total": self.total,
                "error": self.error, "segundos": round(time.time() - self.creado, 1)}


def ejecutar_trabajo(trabajo, config):
    # Graba las réplicas animadas y, si se pidió, corre las réplicas de la estadística en varios procesos
    try:
        trabajo.estado = 'ejecutando'
        escalar_semilla = config["escalar_semilla"]
        escalar_animadas = config["escalar_replicas_animadas"]
        escalar_estadistica = config["escalar_cantidad_replicas"]
        trabajo.total = escalar_animadas + escalar_estadistica

        trabajo.mensaje = 'Grabando la animación'
        datos = generar_datos(config, [escalar_semilla + i for i in range(escalar_animadas)], al_avanzar=trabajo.avanzar)

        estadistica = {"replicas": 0, "politicas": []}
        if escalar_estadistica:
            trabajo.mensaje = 'Calculando la estadística'
            vector_argumentos = [(config, escalar_semilla + i) for i in range(escalar_estadistica)]
            vector_resultados = []
            with mp.Pool(processes=max(1, min(mp.cpu_count(), escalar_estadistica))) as bahia:
                try:
                    for resultado_replica in bahia.imap_unordered(Simulacion.cargarReplica, vector_argumentos):
                        vector_resultados.append(resultado_replica)
                        trabajo.avanzar()
                except Cancelado:
                    bahia.terminate()
                    raise
            estadistica = resumir(vector_resultados)

        trabajo.resultado = {"datos": datos, "estadistica": estadistica, "configuracion": config_a_json(config)}
        trabajo.estado = 'listo'
        trabajo.mensaje = 'Listo'
    except Cancelado:
        trabajo.estado = 'cancelado'
        trabajo.mensaje = 'Cancelado'
    except Exception as error:  # el error llega a la página como mensaje, el detalle queda en la consola del servidor
        trabajo.estado = 'error'
        trabajo.error = f'{type(error).__name__}: {error}'
        import traceback
        traceback.print_exc()


class Estado:
    # Trabajos conocidos por el servidor
    def __init__(self):
        self.trabajos = {}
        self.cerrojo = threading.Lock()

    def crear(self, config):
        with self.cerrojo:
            if any(t.estado in ('en_cola', 'ejecutando') for t in self.trabajos.values()):
                raise RuntimeError('Ya hay una simulación en curso. Espera a que termine o cancélala.')
            trabajo = Trabajo()
            self.trabajos[trabajo.id] = trabajo
            terminados = [t for t in sorted(self.trabajos.values(), key=lambda t: t.creado) if t.estado not in ('en_cola', 'ejecutando')]
            for antiguo in terminados[:-TRABAJOS_GUARDADOS]:
                del self.trabajos[antiguo.id]
        threading.Thread(target=ejecutar_trabajo, args=(trabajo, config), daemon=True).start()
        return trabajo

    def buscar(self, txt_id):
        return self.trabajos.get(txt_id)

    def activo(self):
        # Trabajo que se está ejecutando, si lo hay: permite que una página recargada se reconecte en lugar de iniciar otro
        with self.cerrojo:
            return next((t for t in self.trabajos.values() if t.estado in ('en_cola', 'ejecutando')), None)


def configuracion_por_defecto():
    # Valores iniciales del formulario: la configuración del proyecto con una estadística corta para que la primera ejecución sea ágil
    config = cargar_configuracion()
    config["escalar_cantidad_replicas"] = 20
    return config_a_json(config)


def crear_manejador(estado):
    class Manejador(BaseHTTPRequestHandler):
        server_version = 'ASRS/1.0'

        def log_message(self, formato, *argumentos):
            pass

        def _responder(self, codigo, cuerpo, txt_tipo):
            datos = cuerpo.encode('utf-8') if isinstance(cuerpo, str) else cuerpo
            try:
                self.send_response(codigo)
                self.send_header('Content-Type', txt_tipo)
                self.send_header('Content-Length', str(len(datos)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(datos)
            except ConnectionError:
                pass   # el navegador cerró la conexión (por ejemplo, al recargar la página), no hay a quién responder

        def _json(self, codigo, objeto):
            self._responder(codigo, json.dumps(objeto, separators=(',', ':')), 'application/json; charset=utf-8')

        def do_GET(self):
            ruta = self.path.split('?')[0]
            if ruta in ('/', '/index.html'):
                self._responder(200, leer_plantilla(), 'text/html; charset=utf-8')
            elif ruta == '/api/config':
                self._json(200, configuracion_por_defecto())
            elif ruta == '/api/activo':
                trabajo = estado.activo()
                self._json(200, {"id": trabajo.id if trabajo else None})
            elif ruta.startswith('/api/trabajo/'):
                trabajo = estado.buscar(ruta.rsplit('/', 1)[-1])
                if trabajo is None:
                    self._json(404, {"error": "Trabajo desconocido."})
                    return
                cuerpo = trabajo.estado_publico()
                if trabajo.estado == 'listo' and 'resultado=1' in self.path:
                    cuerpo["resultado"] = trabajo.resultado
                self._json(200, cuerpo)
            else:
                self._json(404, {"error": "No encontrado."})

        def do_POST(self):
            ruta = self.path.split('?')[0]
            if ruta == '/api/simular':
                try:
                    escalar_largo = int(self.headers.get('Content-Length', '0'))
                    if escalar_largo <= 0 or escalar_largo > LIMITE_CUERPO:
                        raise ValueError('La solicitud está vacía o es demasiado grande.')
                    solicitud = json.loads(self.rfile.read(escalar_largo).decode('utf-8'))
                    config = config_desde_json(solicitud.get('config', {}))   # valida antes de crear el trabajo
                    trabajo = estado.crear(config)
                    self._json(202, {"id": trabajo.id})
                except RuntimeError as error:
                    self._json(409, {"error": str(error)})
                except (ValueError, KeyError) as error:
                    self._json(400, {"error": str(error)})
            elif ruta == '/api/tareas':
                # Valida un archivo de tareas reales y devuelve su resumen, para mostrarlo antes de simular
                try:
                    escalar_largo = int(self.headers.get('Content-Length', '0'))
                    if escalar_largo <= 0 or escalar_largo > LIMITE_CUERPO:
                        raise ValueError('La solicitud está vacía o es demasiado grande.')
                    solicitud = json.loads(self.rfile.read(escalar_largo).decode('utf-8'))
                    matriz_tareas = leer_tareas(str(solicitud.get('texto', '')), LIMITE_TAREAS)
                    resumen = resumir_tareas(matriz_tareas, float(solicitud.get('porcentaje', 100)))
                    if resumen["skus"] > LIMITE_SKUS:
                        raise ValueError(f'Datos reales: hay {resumen["skus"]} SKUs distintos, el máximo es {LIMITE_SKUS}.')
                    self._json(200, resumen)
                except (ValueError, KeyError, TypeError) as error:
                    self._json(400, {"error": str(error)})
            elif ruta.startswith('/api/cancelar/'):
                trabajo = estado.buscar(ruta.rsplit('/', 1)[-1])
                if trabajo is None:
                    self._json(404, {"error": "Trabajo desconocido."})
                else:
                    trabajo.cancelado = True
                    self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "No encontrado."})

    return Manejador


class ServidorLocal(ThreadingHTTPServer):
    # En Windows, reutilizar la dirección permite que dos procesos escuchen el mismo puerto y las peticiones caen en uno u otro al azar
    # Por eso en Windows el puerto se reserva en exclusiva: un segundo servidor falla con un error claro en lugar de convivir con el primero
    allow_reuse_address = not hasattr(socket, 'SO_EXCLUSIVEADDRUSE')

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def crear_servidor(escalar_puerto=8765):
    # Servidor listo para usar; con puerto 0 el sistema elige uno libre
    estado = Estado()
    servidor = ServidorLocal(('127.0.0.1', escalar_puerto), crear_manejador(estado))
    servidor.daemon_threads = True
    servidor.estado_trabajos = estado
    return servidor


def servir(escalar_puerto=8765, booleano_abrir=True):
    try:
        servidor = crear_servidor(escalar_puerto)
    except OSError:
        raise SystemExit(f'El puerto {escalar_puerto} ya está en uso, probablemente por otra herramienta abierta.\n'
                         f'Ciérrala con Ctrl+C en su terminal, o usa otro puerto: python visualizar.py --servir --puerto {escalar_puerto + 1}')
    txt_url = f'http://127.0.0.1:{servidor.server_address[1]}/'
    print(f'Herramienta de simulación en {txt_url}  (Ctrl+C para detener)')
    if booleano_abrir:
        threading.Timer(0.5, lambda: webbrowser.open(txt_url)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print('\nServidor detenido.')
    finally:
        servidor.server_close()
