// ================= vista Entrenamiento: cómo aprende la red neuronal =================
// Entrena DQN o PPO en el servidor (proceso aparte) y muestra en vivo la curva de aprendizaje, las pérdidas y la red por dentro:
// lo que ve (la estantería), cómo se activan sus neuronas, qué decide (un mapa de canales) y cómo cambian sus pesos.
(function () {
  const NOMBRES_CLASICAS = { aleatoria: 'Aleatoria', basada_en_clases: 'Basada en clases', full_turnover: 'Full-turnover', mas_cercana_es: 'Más cercana al E/S', mas_cercana_grua: 'Más cercana a la grúa' };
  const GRUPOS = ['Llenado', 'Cercanía al E/S', 'Mismo SKU', 'Con espacio', 'Escalares'];
  let construida = false, actual = null, ultimoN = 0, temporizadorEnt = null;
  let metricas = [], fotos = [], referencias = {}, evaluacion = {}, estadoEnt = 'inactivo', inicio = null, indiceFoto = 0, enVivo = true, finInfo = null, errorEnt = null;
  let zonasRed = [];

  // ---------- utilidades de color y lienzo ----------
  const rgb = h => { const m = /^#?([0-9a-f]{6})$/i.exec(h.trim()); if (!m) return [128, 128, 128]; const n = parseInt(m[1], 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
  const mezclar = (a, b, t) => { const A = rgb(a), B = rgb(b); return 'rgb(' + A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(',') + ')'; };
  const secuencial = t => colorRotacion(1 - Math.max(0, Math.min(1, t)));                // 1 = amarillo (alto), 0 = violeta (bajo)
  const divergente = v => v >= 0 ? mezclar(css('--vacio'), '#f472b6', Math.min(1, v)) : mezclar(css('--vacio'), '#38bdf8', Math.min(1, -v));
  function lienzo(id, alto) {
    const cv = el(id), dpr = Math.min(window.devicePixelRatio || 1, 2), w = cv.clientWidth || 600;
    cv.width = Math.round(w * dpr); cv.height = Math.round(alto * dpr); cv.style.height = alto + 'px';
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, alto);
    return { g, w, h: alto };
  }
  const texto = (g, s, x, y, o = {}) => { g.fillStyle = o.color || css('--suave'); g.font = (o.peso || 400) + ' ' + (o.tam || 11) + 'px system-ui, sans-serif'; g.textAlign = o.alin || 'left'; g.textBaseline = o.base || 'alphabetic'; g.fillText(s, x, y); };

  // ---------- gráfico de líneas ----------
  function lineas(id, alto, series, refs, opc = {}) {
    const { g, w, h } = lienzo(id, alto), mi = { l: 52, r: opc.margenDer || 12, t: 10, b: 24 };
    const puntos = series.flatMap(s => s.puntos), xs = puntos.map(p => p[0]);
    if (!puntos.length) { texto(g, opc.vacio || 'Esperando datos…', w / 2, h / 2, { alin: 'center' }); return; }
    const x0 = 0, x1 = Math.max(opc.xMax || 0, ...xs, 1);
    let ys = puntos.map(p => p[1]).concat((refs || []).map(r => r.y)); let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (y0 === y1) { y0 -= 1; y1 += 1; } const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
    const X = x => mi.l + (x - x0) / (x1 - x0) * (w - mi.l - mi.r), Y = y => h - mi.b - (y - y0) / (y1 - y0) * (h - mi.t - mi.b);
    g.strokeStyle = css('--borde'); g.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = y0 + (y1 - y0) * i / 4; g.beginPath(); g.moveTo(mi.l, Y(y)); g.lineTo(w - mi.r, Y(y)); g.stroke();
      texto(g, opc.formato ? opc.formato(y) : fmt(y), mi.l - 6, Y(y) + 4, { alin: 'right' });
    }
    for (let i = 0; i <= 4; i++) texto(g, fmt(x0 + (x1 - x0) * i / 4), X(x0 + (x1 - x0) * i / 4), h - 6, { alin: i === 0 ? 'left' : (i === 4 ? 'right' : 'center') });
    (refs || []).forEach(r => {
      g.strokeStyle = r.color; g.lineWidth = 1.2; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(mi.l, Y(r.y)); g.lineTo(w - mi.r, Y(r.y)); g.stroke(); g.setLineDash([]);
      if (r.etiqueta) texto(g, r.etiqueta, w - mi.r + 4, Y(r.y) + 4, { color: r.color, tam: 10 });
    });
    series.forEach(s => {
      g.strokeStyle = s.color; g.lineWidth = 2; g.lineJoin = 'round'; g.beginPath();
      s.puntos.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))); g.stroke();
    });
  }

  // ---------- mapa de calor de una estantería (filas x columnas) o de una cuadrícula ----------
  function mapa(g, valores, filas, cols, x, y, w, h, color, opc = {}) {
    const cw = w / cols, ch = h / filas;
    for (let f = 0; f < filas; f++) for (let c = 0; c < cols; c++) {
      const v = valores[f * cols + c];
      g.fillStyle = v === null || v === undefined ? (opc.vacio || css('--hueco')) : color(v);
      const yy = opc.abajo ? y + h - (f + 1) * ch : y + f * ch;                      // con 'abajo' la fila 0 queda abajo, como en la estantería
      g.fillRect(x + c * cw, yy, Math.max(1, cw - (cw > 5 ? 1 : 0)), Math.max(1, ch - (ch > 5 ? 1 : 0)));
    }
    g.strokeStyle = css('--borde'); g.lineWidth = 1; g.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);
  }

  // ---------- la red por dentro ----------
  function dibujarRed() {
    const foto = fotos[indiceFoto]; zonasRed = [];
    if (!foto || !inicio) { lienzo('cvRed', 120); texto(lienzo('cvRed', 120).g, 'Aquí aparecerá la red cuando empiece el entrenamiento.', 20, 60); return; }
    const filas = inicio.filas, cols = inicio.columnas, alto = 470, { g, w } = lienzo('cvRed', alto);
    const colEntrada = 20, anchoMini = Math.min(150, w * 0.17), altoMini = anchoMini * filas / cols * 0.78;
    // 1. Entrada: cuatro mapas de la estantería que ve el agente
    texto(g, 'Lo que ve', colEntrada, 14, { color: css('--texto'), peso: 700, tam: 12 });
    const yMini = i => 26 + i * (altoMini + 30);
    foto.entrada.forEach((v, i) => {
      texto(g, GRUPOS[i], colEntrada, yMini(i) + 9, { tam: 10 });
      mapa(g, v, filas, cols, colEntrada, yMini(i) + 14, anchoMini, altoMini, t => secuencial(t / 255), { abajo: true });
    });
    texto(g, 'rango del SKU ' + fmt1(foto.escalares[0] * 100) + ' % · avance ' + fmt1(foto.escalares[1] * 100) + ' %', colEntrada, yMini(3) + altoMini + 30, { tam: 10 });

    // 2. Capas ocultas: una cuadrícula de neuronas coloreadas por su activación
    const xCap = [w * 0.34, w * 0.52], lado = Math.min(w * 0.14, 200), pesos = foto.pesos;
    const nNeur = foto.activaciones.map(a => a.length);
    foto.activaciones.forEach((a, k) => {
      const n = a.length, lados = Math.ceil(Math.sqrt(n)), maxAbs = Math.max(1e-9, ...a.map(Math.abs)), hayNeg = a.some(v => v < -1e-6);
      const valores = a.map(v => hayNeg ? v / maxAbs : v / maxAbs);
      const y = 40, cel = lado / lados;
      texto(g, 'Capa oculta ' + (k + 1), xCap[k], 14, { color: css('--texto'), peso: 700, tam: 12 });
      texto(g, n + ' neuronas', xCap[k], 28, { tam: 10 });
      for (let i = 0; i < n; i++) {
        const f = Math.floor(i / lados), c = i % lados;
        g.fillStyle = hayNeg ? divergente(valores[i]) : secuencial(valores[i]);
        g.fillRect(xCap[k] + c * cel, y + 14 + f * cel, Math.max(1, cel - 1), Math.max(1, cel - 1));
      }
      zonasRed.push({ x: xCap[k], y: y + 14, w: lados * cel, h: lados * cel, lados, cel, capa: k, valores: a });
      g.strokeStyle = css('--borde'); g.strokeRect(xCap[k] - 0.5, y + 13.5, lados * cel + 1, lados * cel + 1);
      texto(g, hayNeg ? 'azul negativa · rosa positiva' : 'violeta apagada · amarilla muy activa', xCap[k], y + 14 + lados * cel + 16, { tam: 10 });
    });

    // 3. Salida: el mapa de canales de la estantería con lo que decide la red
    const xSal = w * 0.72, anchoSal = Math.min(w * 0.25, 260), altoSal = anchoSal * filas / cols * 0.8, sal = foto.salida;
    texto(g, 'Lo que decide', xSal, 14, { color: css('--texto'), peso: 700, tam: 12 });
    texto(g, sal.tipo === 'prob' ? 'probabilidad de elegir cada canal' : 'valor Q de cada canal', xSal, 28, { tam: 10 });
    const numeros = sal.valores.filter(v => v !== null), mn = Math.min(...numeros), mx = Math.max(...numeros), rango = Math.max(mx - mn, 1e-9);
    mapa(g, sal.valores.map(v => v === null ? null : (v - mn) / rango), filas, cols, xSal, 40, anchoSal, altoSal, secuencial, { abajo: true });
    const fe = Math.floor(sal.elegida / cols), ce = sal.elegida % cols;
    g.strokeStyle = css('--texto'); g.lineWidth = 2; g.strokeRect(xSal + ce * anchoSal / cols - 1, 40 + altoSal - (fe + 1) * altoSal / filas - 1, anchoSal / cols + 2, altoSal / filas + 2);
    texto(g, 'Canal elegido: fila ' + (fe + 1) + ', columna ' + (ce + 1), xSal, 40 + altoSal + 16, { tam: 11, color: css('--texto') });
    texto(g, 'gris: canales sin espacio', xSal, 40 + altoSal + 30, { tam: 10 });

    // 4. Flechas entre etapas, con el tamaño medio de los pesos de cada capa
    const flecha = (x1, x2, y, k) => {
      const p = pesos[k]; if (!p) return;
      g.strokeStyle = css('--suave'); g.lineWidth = 1 + Math.min(5, p.media_abs * 25); g.globalAlpha = 0.55;
      g.beginPath(); g.moveTo(x1, y); g.lineTo(x2 - 6, y); g.stroke(); g.globalAlpha = 1;
      g.fillStyle = css('--suave'); g.beginPath(); g.moveTo(x2, y); g.lineTo(x2 - 9, y - 5); g.lineTo(x2 - 9, y + 5); g.closePath(); g.fill();
      texto(g, 'pesos ' + p.entrada + '→' + p.salida, (x1 + x2) / 2, y - 8, { alin: 'center', tam: 10 });
      texto(g, 'media |w| ' + p.media_abs.toFixed(3), (x1 + x2) / 2, y + 16, { alin: 'center', tam: 10 });
    };
    const yMed = 40 + lado / 2 + 14;
    flecha(colEntrada + anchoMini + 8, xCap[0] - 6, yMed, 0);
    flecha(xCap[0] + lado + 6, xCap[1] - 6, yMed, 1);
    flecha(xCap[1] + lado + 6, xSal - 6, yMed, 2);
    zonasRed.forEach(z => { z.alturaTotal = alto; });
  }

  // ---------- pesos: histogramas, atención de la primera capa y mapa de la segunda ----------
  function dibujarPesos() {
    const foto = fotos[indiceFoto]; if (!foto) { lienzo('cvHist', 120); lienzo('cvAtencion', 120); lienzo('cvMapaPesos', 120); return; }
    // histogramas de cada capa de pesos
    {
      const { g, w } = lienzo('cvHist', 170), n = foto.pesos.length, an = (w - 20) / n;
      foto.pesos.forEach((p, k) => {
        const x0 = 10 + k * an, maximo = Math.max(...p.hist, 1), bw = (an - 24) / p.hist.length;
        texto(g, 'Capa ' + (k + 1) + ' (' + p.entrada + '→' + p.salida + ')', x0 + 4, 14, { color: css('--texto'), peso: 600, tam: 11 });
        p.hist.forEach((c, i) => { g.fillStyle = css('--acento'); const hh = c / maximo * 100; g.fillRect(x0 + 4 + i * bw, 130 - hh, Math.max(1, bw - 1), hh); });
        g.strokeStyle = css('--borde'); g.beginPath(); g.moveTo(x0 + 4, 130.5); g.lineTo(x0 + 4 + bw * p.hist.length, 130.5); g.stroke();
        texto(g, p.min.toFixed(2), x0 + 4, 144, { tam: 10 }); texto(g, p.max.toFixed(2), x0 + 4 + bw * p.hist.length, 144, { tam: 10, alin: 'right' });
        texto(g, 'desv. ' + p.desv.toFixed(3), x0 + 4, 160, { tam: 10 });
      });
    }
    // atención de cada neurona de la capa 1 a cada grupo de entradas
    {
      const p = foto.pesos[0], grupos = p.grupos, n = p.salida, lados = Math.ceil(Math.sqrt(n)), { g, w } = lienzo('cvAtencion', 170);
      const maximo = Math.max(...p.atencion), an = (w - 20) / grupos, lado = Math.min(an - 14, 120), cel = lado / lados;
      for (let k = 0; k < grupos; k++) {
        const x0 = 10 + k * an;
        texto(g, GRUPOS[k] || 'Grupo ' + (k + 1), x0, 14, { color: css('--texto'), peso: 600, tam: 11 });
        for (let i = 0; i < n; i++) {
          g.fillStyle = secuencial(p.atencion[i * grupos + k] / maximo);
          g.fillRect(x0 + (i % lados) * cel, 22 + Math.floor(i / lados) * cel, Math.max(1, cel - 0.5), Math.max(1, cel - 0.5));
        }
      }
    }
    // mapa de los pesos entre las dos capas ocultas y peso medio hacia cada canal de salida
    {
      const { g, w } = lienzo('cvMapaPesos', 190), p = foto.pesos[1], s = foto.pesos[foto.pesos.length - 1];
      const m1 = Math.max(...p.pool.map(Math.abs), 1e-9), lado = 150;
      texto(g, 'Entre las capas ocultas', 10, 14, { color: css('--texto'), peso: 600, tam: 11 });
      mapa(g, p.pool.map(v => v / m1), 32, 32, 10, 24, lado, lado, divergente);
      const m2 = Math.max(...s.por_canal.map(Math.abs), 1e-9), x2 = 10 + lado + 40, anchoS = Math.min(w - x2 - 10, 190);
      texto(g, 'Peso medio hacia cada canal', x2, 14, { color: css('--texto'), peso: 600, tam: 11 });
      mapa(g, s.por_canal.map(v => v / m2), inicio.filas, inicio.columnas, x2, 24, anchoS, anchoS * inicio.filas / inicio.columnas * 0.8, divergente, { abajo: true });
    }
  }

  // ---------- curvas ----------
  function dibujarCurvas() {
    const series = [{ color: css('--acento'), puntos: metricas.filter(m => m.tiempo !== null && m.tiempo !== undefined).map(m => [m.pasos, m.tiempo]) }];
    const colores = ['#94a3b8', '#fbbf24', '#c084fc', '#2dd4bf', '#f472b6'];
    const refs = Object.keys(NOMBRES_CLASICAS).filter(k => referencias[k] !== undefined).map((k, i) => ({ y: referencias[k], color: colores[i], etiqueta: NOMBRES_CLASICAS[k] }));
    Object.keys(evaluacion).forEach(k => refs.push({ y: evaluacion[k], color: css('--texto'), etiqueta: k + ' (prueba)' }));
    lineas('cvCurva', 250, series, refs, { margenDer: 120, xMax: inicio ? inicio.pasos : 0, formato: v => fmt(v) + ' s', vacio: 'La curva aparece cuando termina el primer episodio.' });
    const clave = el('selPerdida').value;
    const puntos = metricas.map(m => [m.pasos, clave === 'epsilon' ? m.epsilon : (m.perdidas || {})[clave]]).filter(p => p[1] !== undefined && p[1] !== null);
    lineas('cvPerdida', 250, [{ color: css('--recuperar'), puntos }], [], { xMax: inicio ? inicio.pasos : 0, formato: v => Math.abs(v) >= 100 ? fmt(v) : v.toPrecision(3), vacio: 'Las pérdidas aparecen tras la primera actualización de la red.' });
  }

  function actualizarSelectorPerdida() {
    const sel = el('selPerdida'), claves = new Set();
    metricas.forEach(m => { Object.keys(m.perdidas || {}).forEach(k => { if (!['learning_rate', 'n_updates', 'clip_range'].includes(k)) claves.add(k); }); if (m.epsilon !== undefined) claves.add('epsilon'); });
    const previa = sel.value, lista = Array.from(claves);
    if (lista.join() !== Array.from(sel.options).map(o => o.value).join()) {
      sel.innerHTML = lista.map(k => '<option value="' + k + '">' + k + '</option>').join('');
      sel.value = lista.includes(previa) ? previa : (lista.includes('loss') ? 'loss' : lista[0]);
    }
  }

  // ---------- estado y texto ----------
  function dibujarEstado() {
    const caja = el('entEstado'), ultima = metricas[metricas.length - 1];
    const activo = ['iniciando', 'referencias', 'entrenando', 'evaluando'].includes(estadoEnt);
    el('btnEntrenar').disabled = activo; el('btnDetener').disabled = !activo;
    let html = '', pct = 0;
    if (estadoEnt === 'inactivo') html = 'Entrena un agente con los parámetros del almacén de la izquierda y observa cómo aprende.';
    else if (estadoEnt === 'iniciando') html = 'Iniciando el proceso de entrenamiento…';
    else if (estadoEnt === 'referencias') { html = 'Calculando cuánto tardan las políticas clásicas en los estados de prueba (' + Object.keys(referencias).length + '/5)…'; }
    else if (estadoEnt === 'entrenando' && ultima) {
      pct = 100 * ultima.pasos / inicio.pasos; const falta = (inicio.pasos - ultima.pasos) / Math.max(ultima.pps, 1e-9);
      html = '<b>' + fmt(ultima.pasos) + '</b> de ' + fmt(inicio.pasos) + ' pasos · ' + fmt(ultima.pps) + ' pasos/s · ' + ultima.episodios + ' episodios · faltan ≈ ' + fmt(falta) + ' s';
    } else if (estadoEnt === 'entrenando') html = 'Entrenando… esperando las primeras mediciones.';
    else if (estadoEnt === 'evaluando') { html = 'Evaluando el agente en estados que no vio…'; pct = 100; }
    else if (estadoEnt === 'listo') {
      pct = 100; const clave = Object.keys(evaluacion)[0], valor = evaluacion[clave], mejor = Object.entries(referencias).sort((a, b) => a[1] - b[1])[0];
      html = '<b>Listo</b> en ' + fmt(finInfo ? finInfo.segundos : 0) + ' s (' + fmt(finInfo ? finInfo.pps : 0) + ' pasos/s). ' + clave + ': ' + fmt(valor) + ' s en los estados de prueba' +
        (mejor ? ', ' + (valor <= mejor[1] ? 'mejor' : 'peor') + ' que la mejor política clásica (' + NOMBRES_CLASICAS[mejor[0]] + ', ' + fmt(mejor[1]) + ' s): ' + (100 * (valor / mejor[1] - 1)).toFixed(1) + ' %.' : '.');
    } else if (estadoEnt === 'cancelado') html = 'Entrenamiento detenido.';
    else if (estadoEnt === 'error') html = '<span style="color:var(--omitida)">El entrenamiento falló.</span><pre class="ent-error">' + (errorEnt || '').replace(/</g, '&lt;') + '</pre>';
    caja.innerHTML = '<div>' + html + '</div>' + (activo ? '<div class="progreso"><div class="barra"><i style="width:' + pct + '%"></i></div></div>' : '');
    const sl = el('slFoto'); sl.max = Math.max(0, fotos.length - 1); sl.value = indiceFoto; el('lblFoto').textContent = fotos.length ? 'foto ' + (indiceFoto + 1) + '/' + fotos.length + ' · paso ' + fmt(fotos[indiceFoto].pasos) : '—';
    el('chkVivo').checked = enVivo;
  }

  function redibujar() { dibujarEstado(); dibujarCurvas(); dibujarRed(); dibujarPesos(); }

  // ---------- comunicación con el servidor ----------
  async function consultarEnt() {
    if (!actual) return;
    try {
      const r = await fetch('/api/entrenamiento/' + actual + '?desde=' + ultimoN), j = await r.json();
      if (!r.ok) throw new Error(j.error || 'Se perdió el entrenamiento.');
      estadoEnt = j.estado; inicio = j.inicio || inicio; referencias = j.referencias || {}; evaluacion = j.evaluacion || {}; finInfo = j.fin; errorEnt = j.error;
      ultimoN = j.secuencia; metricas = metricas.concat(j.metricas);
      if (j.fotos.length) { fotos = fotos.concat(j.fotos); if (enVivo) indiceFoto = fotos.length - 1; }
      actualizarSelectorPerdida(); redibujar();
      if (['listo', 'error', 'cancelado'].includes(j.estado)) return;
      temporizadorEnt = setTimeout(consultarEnt, 1000);
    } catch (e) { estadoEnt = 'error'; errorEnt = e.message; redibujar(); }
  }

  async function entrenar() {
    let cfg; try { cfg = leerFormulario(); } catch (e) { aviso('error', e.message); return; }
    ocultarAviso();
    const cuerpo = { config: cfg, algoritmo: el('selAlgoritmo').value, pasos: Number(el('inPasos').value), entornos: Number(el('inEntornos').value), estados: 16, pruebas: 4 };
    try {
      const r = await fetch('/api/entrenar', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cuerpo) }), j = await r.json();
      if (!r.ok) throw new Error(j.error || 'No se pudo iniciar el entrenamiento.');
      reiniciarDatos(); actual = j.id; estadoEnt = 'iniciando'; redibujar(); consultarEnt();
    } catch (e) { aviso('error', e.message); }
  }
  function reiniciarDatos() { clearTimeout(temporizadorEnt); metricas = []; fotos = []; referencias = {}; evaluacion = {}; ultimoN = 0; indiceFoto = 0; enVivo = true; inicio = null; finInfo = null; errorEnt = null; el('selPerdida').innerHTML = ''; }

  // ---------- construcción de la página ----------
  function construir() {
    if (construida) return; construida = true;
    const vista = el('vistaEntrenamiento');
    vista.innerHTML =
      '<section class="ent-tarjeta"><div class="ent-controles">' +
      '<label>Algoritmo <select id="selAlgoritmo"><option value="ppo">PPO</option><option value="dqn">DQN</option></select></label>' +
      '<label>Pasos <input type="number" id="inPasos" min="1000" max="2000000" step="1000" value="20000"></label>' +
      '<label title="Cada entorno corre en un núcleo del procesador">Entornos en paralelo <input type="number" id="inEntornos" min="1" max="64" step="1"></label>' +
      '<button id="btnEntrenar" class="primario">Entrenar</button><button id="btnDetener" disabled>Detener</button></div>' +
      '<div id="entEstado" class="ent-estado"></div>' +
      '<p class="nota">El agente aprende en qué canal almacenar cada ítem. Usa el almacén, la demanda y las tareas de los parámetros de la izquierda. Se compara con las políticas clásicas en estados de prueba que no ve al entrenar.</p></section>' +
      '<div class="ent-cuadricula"><section class="ent-tarjeta"><h3>Curva de aprendizaje</h3><p class="nota">Tiempo total de la lista de tareas por episodio (media de los últimos 30; menos es mejor). Las líneas punteadas son las políticas clásicas.</p><canvas id="cvCurva"></canvas></section>' +
      '<section class="ent-tarjeta"><h3>Pérdidas <select id="selPerdida"></select></h3><p class="nota">Cuánto se equivoca la red en cada actualización. En DQN, epsilon es la parte de decisiones al azar (exploración).</p><canvas id="cvPerdida"></canvas></section></div>' +
      '<section class="ent-tarjeta"><div class="ent-cabecera"><h3>La red por dentro</h3><div class="ent-replay"><label><input type="checkbox" id="chkVivo" checked> En vivo</label><input type="range" id="slFoto" min="0" max="0" value="0"><span id="lblFoto">—</span></div></div>' +
      '<p class="nota">La red mira siempre la misma situación de la estantería y, con el avance del entrenamiento, cambia lo que decide. Desliza para comparar el principio con el final. Pasa el ratón sobre una neurona para ver su activación.</p>' +
      '<canvas id="cvRed"></canvas><div id="entTip" class="nota">&nbsp;</div></section>' +
      '<div class="ent-cuadricula"><section class="ent-tarjeta"><h3>Distribución de los pesos</h3><p class="nota">Cada barra cuenta cuántos pesos tienen ese valor. Al aprender, las distribuciones se ensanchan o se desplazan.</p><canvas id="cvHist"></canvas></section>' +
      '<section class="ent-tarjeta"><h3>Pesos entre capas y hacia los canales</h3><canvas id="cvMapaPesos"></canvas></section></div>' +
      '<section class="ent-tarjeta"><h3>A qué mira cada neurona de la primera capa</h3><p class="nota">Cada cuadrícula tiene una neurona por casilla; el color indica cuánto pesa en ella ese grupo de entradas. Amarillo es mucha atención.</p><canvas id="cvAtencion"></canvas></section>';
    el('inEntornos').value = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
    el('btnEntrenar').onclick = entrenar;
    el('btnDetener').onclick = () => { if (actual) fetch('/api/entrenamiento/' + actual + '/cancelar', { method: 'POST' }); };
    el('selPerdida').onchange = dibujarCurvas;
    el('slFoto').oninput = () => { indiceFoto = +el('slFoto').value; enVivo = indiceFoto === fotos.length - 1; dibujarEstado(); dibujarRed(); dibujarPesos(); };
    el('chkVivo').onchange = () => { enVivo = el('chkVivo').checked; if (enVivo && fotos.length) { indiceFoto = fotos.length - 1; redibujar(); } };
    el('cvRed').addEventListener('mousemove', ev => {
      const r = el('cvRed').getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
      const z = zonasRed.find(z => x >= z.x && x < z.x + z.w && y >= z.y && y < z.y + z.h);
      if (!z) { el('entTip').innerHTML = '&nbsp;'; return; }
      const i = Math.floor((y - z.y) / z.cel) * z.lados + Math.floor((x - z.x) / z.cel);
      el('entTip').textContent = i < z.valores.length ? 'Capa oculta ' + (z.capa + 1) + ' · neurona ' + (i + 1) + ' · activación ' + z.valores[i].toFixed(3) : '';
    });
    window.addEventListener('resize', () => { if (!el('vistaEntrenamiento').hidden) redibujar(); });
    redibujar();
  }

  async function reconectar() {
    if (actual) return;
    try {
      const id = (await (await fetch('/api/entrenamiento/activo')).json()).id;
      if (id) { reiniciarDatos(); actual = id; estadoEnt = 'iniciando'; consultarEnt(); }
    } catch (e) {}
  }

  // ---------- cambio entre la carrera y el entrenamiento ----------
  function mostrar(nombre) {
    document.querySelectorAll('#navVistas button').forEach(b => b.classList.toggle('activa', b.dataset.v === nombre));
    el('vistaCarrera').hidden = nombre !== 'carrera'; el('vistaEntrenamiento').hidden = nombre !== 'entrenamiento';
    if (nombre === 'entrenamiento') { construir(); reconectar(); setTimeout(redibujar, 0); }
    try { localStorage.setItem('asrs_vista', nombre); } catch (e) {}
  }
  document.querySelectorAll('#navVistas button').forEach(b => { b.onclick = () => mostrar(b.dataset.v); });
  window.mostrarVistaApp = mostrar;
})();
