// VicCount — interfața. Modelele rulează în worker.js, pe telefon.
const $ = (id) => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d'), stage = $('stage');
const MAX = 1600;
let img = null, bitmap = null, ex = [], res = null, removed = new Set(), added = [], drag = null;

// ---------------- service worker (offline + multi-thread)
async function setupSW() {
  if (!('serviceWorker' in navigator)) return;
  try {
    await navigator.serviceWorker.register('sw.js');
    if (!navigator.serviceWorker.controller && !sessionStorage.getItem('vc-reload')) {
      sessionStorage.setItem('vc-reload', '1');
      await navigator.serviceWorker.ready;
      location.reload();
      return new Promise(() => {});
    }
  } catch (e) { console.warn('SW', e); }
}

// ---------------- worker
let worker, seq = 0;
const pending = new Map();
function call(msg, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = ++seq; pending.set(id, { resolve, reject });
    worker.postMessage({ ...msg, id }, transfer);
  });
}
function startWorker() {
  worker = new Worker('worker.js', { type: 'module' });
  const files = {};
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'progress') {
      files[m.file] = [m.loaded, m.total];
      const [l, t] = Object.values(files).reduce((a, v) => [a[0] + v[0], a[1] + v[1]], [0, 0]);
      busy(true, `Descarc modelul (o singură dată)… ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB`, l / t);
    } else if (m.type === 'stage') {
      $('busyTxt').textContent = m.text;
    } else if (m.type === 'ready') {
      pending.get(m.id)?.resolve(m); pending.delete(m.id);
    } else if (m.type === 'result') {
      pending.get(m.id)?.resolve(m.res); pending.delete(m.id);
    } else if (m.type === 'error') {
      pending.get(m.id)?.reject(new Error(m.message)); pending.delete(m.id);
    }
  };
  worker.onerror = (e) => { $('status').innerHTML = '<span class="err">eroare motor</span>'; console.error(e); };
}

function busy(on, text, frac) {
  $('busy').classList.toggle('on', on);
  if (text) $('busyTxt').textContent = text;
  $('bar').hidden = frac == null;
  if (frac != null) $('barFill').style.width = Math.round(frac * 100) + '%';
}

// ---------------- RO -> EN (detectorul înțelege engleză)
const RO = {
  sticla:'bottle',sticle:'bottle',pahar:'glass',pahare:'glass',cana:'cup',cani:'cup',ceasca:'cup',
  om:'person',oameni:'person',persoana:'person',persoane:'person',copil:'child',copii:'child',
  masina:'car',masini:'car',auto:'car',camion:'truck',bicicleta:'bicycle',motocicleta:'motorcycle',
  scaun:'chair',scaune:'chair',masa:'table',mese:'table',sezlong:'sunbed',sezlonguri:'sunbed',umbrela:'umbrella',umbrele:'umbrella',
  moneda:'coin',monede:'coin',banut:'coin',banuti:'coin',bancnota:'banknote',
  ou:'egg',oua:'egg',mar:'apple',mere:'apple',portocala:'orange',portocale:'orange',lamaie:'lemon',rosie:'tomato',rosii:'tomato',
  cutie:'box',cutii:'box',palet:'pallet',paleti:'pallet',sac:'bag',saci:'bag',punga:'bag',
  surub:'screw',suruburi:'screw',piulita:'nut',cui:'nail',cuie:'nail',
  caramida:'brick',caramizi:'brick',teava:'pipe',tevi:'pipe',busteni:'log',bustean:'log',
  cablu:'cable',cabluri:'cable',microfon:'microphone',microfoane:'microphone',boxa:'speaker',boxe:'speaker',
  pasare:'bird',pasari:'bird',oaie:'sheep',oi:'sheep',vaca:'cow',vaci:'cow',caine:'dog',caini:'dog',pisica:'cat',pisici:'cat',peste:'fish',pesti:'fish',
  floare:'flower',flori:'flower',copac:'tree',copaci:'tree',frunza:'leaf',frunze:'leaf',
  pastila:'pill',pastile:'pill',creion:'pencil',creioane:'pencil',pix:'pen',pixuri:'pen',carte:'book',carti:'book',
  bec:'light bulb',becuri:'light bulb',dop:'bottle cap',dopuri:'bottle cap',capac:'lid',capace:'lid',
  doza:'can',doze:'can',farfurie:'plate',farfurii:'plate',lingura:'spoon',furculita:'fork',cutit:'knife',
  telefon:'phone',telefoane:'phone',laptop:'laptop',fereastra:'window',ferestre:'window',usa:'door',usi:'door',
  barca:'boat',barci:'boat',casa:'house',case:'house',minge:'ball',mingi:'ball',
};
function toEnglish(t) {
  const n = t.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').trim();
  return n.split(/\s+/).map((w) => RO[w] || w).join(' ');
}

// ---------------- încărcare imagine
stage.addEventListener('click', () => { if (!img) $('file').click(); });
$('pick').onclick = () => $('file').click();
$('file').onchange = (e) => e.target.files[0] && load(e.target.files[0]);
window.addEventListener('paste', (e) => { const f = [...e.clipboardData.files][0]; f && load(f); });

async function load(file) {
  const b = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const s = Math.min(1, MAX / Math.max(b.width, b.height));
  const c = document.createElement('canvas');
  c.width = Math.round(b.width * s); c.height = Math.round(b.height * s);
  c.getContext('2d').drawImage(b, 0, 0, c.width, c.height);
  img = c; cv.width = c.width; cv.height = c.height;
  ex = []; res = null; removed.clear(); added = [];
  stage.classList.add('has'); $('go').disabled = false; $('count').textContent = '–'; $('meta').textContent = '';
  $('save').disabled = true; $('clr').disabled = true; $('file').value = '';
  draw();
  run();
}

// ---------------- desen
function px(e) { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; }
function total() { return res ? Math.max(0, res.count - removed.size + added.length) : 0; }
function draw() {
  if (!img) return;
  ctx.drawImage(img, 0, 0);
  const W = cv.width, lw = Math.max(2, W / 450), rad = Math.max(6, W / 160);
  const dot = (x, y, col) => { ctx.beginPath(); ctx.arc(x, y, rad, 0, 7); ctx.fillStyle = col; ctx.fill(); ctx.lineWidth = lw * 0.8; ctx.strokeStyle = '#000'; ctx.stroke(); };
  if (res) {
    res.points.forEach((p, i) => dot(p[0], p[1], removed.has(i) ? '#ff5d5d' : '#2ee6a6'));
    added.forEach((p) => dot(p[0], p[1], '#5db8ff'));
    $('count').textContent = total();
  }
  ctx.lineWidth = lw * 1.5; ctx.strokeStyle = '#ffb020'; ctx.setLineDash([lw * 4, lw * 3]);
  for (const b of ex) ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
  if (drag) ctx.strokeRect(drag[0], drag[1], drag[2] - drag[0], drag[3] - drag[1]);
  ctx.setLineDash([]);
}

cv.onpointerdown = (e) => { const [x, y] = px(e); drag = [x, y, x, y]; cv.setPointerCapture(e.pointerId); };
cv.onpointermove = (e) => { if (!drag) return; const [x, y] = px(e); drag[2] = x; drag[3] = y; draw(); };
cv.onpointerup = () => {
  if (!drag) return;
  const [x0, y0, x1, y1] = drag; drag = null;
  const b = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
  const minSide = Math.max(10, cv.width / 80);
  if (b[2] - b[0] > minSide && b[3] - b[1] > minSide) { ex.push(b); $('clr').disabled = false; draw(); return; }
  if (!res) return draw();
  const lim = Math.max(18, cv.width / 40);
  // atingere: scoate/repune un punct, șterge unul adăugat, sau adaugă
  let best = -1, bd = 1e9;
  res.points.forEach((p, i) => { const d = Math.hypot(p[0] - x0, p[1] - y0); if (d < bd) { bd = d; best = i; } });
  const ai = added.findIndex((p) => Math.hypot(p[0] - x0, p[1] - y0) < lim);
  if (ai >= 0) added.splice(ai, 1);
  else if (best >= 0 && bd < lim) removed.has(best) ? removed.delete(best) : removed.add(best);
  else added.push([x0, y0]);
  draw();
};
$('clr').onclick = () => { ex = []; $('clr').disabled = true; draw(); };

// ---------------- numărare
$('go').onclick = run;
$('text').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.target.blur(); run(); } });

async function run() {
  if (!img) return;
  busy(true, ex.length ? 'Număr după exemplu…' : 'Număr…'); $('go').disabled = true;
  try {
    const bm = await createImageBitmap(img);
    const r = await call({ cmd: 'count', bitmap: bm, W: img.width, H: img.height, text: toEnglish($('text').value), boxes: ex }, [bm]);
    res = r; removed.clear(); added = [];
    $('meta').textContent = `${r.mode} · ${r.seconds.toFixed(1)} s`;
    $('save').disabled = false;
  } catch (err) {
    $('meta').innerHTML = '<span class="err">Eroare: ' + String(err.message).slice(0, 160) + '</span>';
  }
  busy(false); $('go').disabled = false; draw();
}

// ---------------- salvare / partajare
$('save').onclick = async () => {
  const c = document.createElement('canvas'); c.width = cv.width; c.height = cv.height;
  const g = c.getContext('2d'); g.drawImage(cv, 0, 0);
  const fs = Math.max(30, c.width / 16), t = 'Total: ' + total();
  g.font = `800 ${fs}px system-ui`; const w = g.measureText(t).width;
  g.fillStyle = '#000b'; g.fillRect(0, 0, w + fs, fs * 1.6); g.fillStyle = '#2ee6a6'; g.fillText(t, fs / 2, fs * 1.15);
  const blob = await new Promise((r) => c.toBlob(r, 'image/jpeg', 0.92));
  const file = new File([blob], `viccount_${total()}_${Date.now()}.jpg`, { type: 'image/jpeg' });
  if (navigator.canShare?.({ files: [file] })) { try { await navigator.share({ files: [file] }); return; } catch {} }
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = file.name; a.click();
};

// ---------------- pornire
(async () => {
  await setupSW();
  startWorker();
  $('status').textContent = 'pregătesc modelul…';
  busy(true, 'Pregătesc modelul…');
  try {
    const r = await call({ cmd: 'warmup' });
    $('status').textContent = `pe telefon · offline ✓ · ${r.threads} fire`;
  } catch (e) {
    $('status').innerHTML = '<span class="err">modelul nu s-a încărcat</span>';
    $('meta').innerHTML = '<span class="err">' + String(e.message).slice(0, 200) + '</span>';
  }
  if (!img) busy(false);
})();
