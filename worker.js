// VicCount — motorul de numărare, rulează în Web Worker, 100% pe dispozitiv.
// Un singur model mic (DINOv2 cu registre, small, int8 ~24 MB) => încape în memoria oricărui telefon.
//  - după exemplu: metoda CountingDINO (WACV 2026): similaritate de trăsături cu exemplul -> vârfuri
//  - automat: separă obiectele de fundal (PCA pe trăsături), alege singur exemplul, apoi la fel
import { AutoModel, Tensor, env } from './vendor/transformers.min.js';

const BASE = new URL('./', self.location.href).href;
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = BASE + 'models/';
env.useBrowserCache = false;                       // cache-ul îl face service worker-ul
env.backends.onnx.wasm.wasmPaths = BASE + 'vendor/';
env.backends.onnx.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const FEAT_ID = 'dinov2-reg-small';
const TILE = 518, PATCH = 14, G = TILE / PATCH;    // 37x37 patch-uri / bucată
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const MAX_TILES = 9;

let featModel = null;

const post = (type, data) => self.postMessage({ type, ...data });
const progress = (label) => (p) => {
  if (p.status === 'progress' && p.total) post('progress', { label, file: p.file, loaded: p.loaded, total: p.total });
};

function loadFeat() {
  if (!featModel) {
    featModel = AutoModel.from_pretrained(FEAT_ID, {
      dtype: 'q8', device: 'wasm', progress_callback: progress('Model'),
    });
  }
  return featModel;
}

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };

// ------------------------------------------------- trăsături DINOv2 pe bucăți
async function featureGrid(bitmap, W, H, exemplars, longTiles = 0) {
  const model = await loadFeat();
  let s;
  if (longTiles) {
    s = (longTiles * TILE) / Math.max(W, H);       // scară fixă (pt. căutarea automată)
  } else {
    // scară: exemplul să ocupe ~4 patch-uri (56px) -> rezoluție bună și pt. obiecte mici
    const exSize = median(exemplars.map((b) => Math.sqrt((b[2] - b[0]) * (b[3] - b[1]))));
    s = Math.max(TILE / Math.max(W, H), 56 / Math.max(exSize, 1));
  }
  let cols = Math.ceil((W * s) / TILE), rows = Math.ceil((H * s) / TILE);
  while (cols * rows > MAX_TILES) { s *= 0.85; cols = Math.ceil((W * s) / TILE); rows = Math.ceil((H * s) / TILE); }
  cols = Math.max(1, cols); rows = Math.max(1, rows);
  const CW = cols * TILE, CH = rows * TILE;
  const sx = CW / W, sy = CH / H;

  const canvas = new OffscreenCanvas(CW, CH);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, CW, CH);

  const GW = cols * G, GH = rows * G;
  let D = 0, F = null;
  const n = rows * cols;
  let done = 0;
  for (let ty = 0; ty < rows; ty++) {
    for (let tx = 0; tx < cols; tx++) {
      const px = ctx.getImageData(tx * TILE, ty * TILE, TILE, TILE).data;
      const arr = new Float32Array(3 * TILE * TILE);
      for (let i = 0, p = 0; i < TILE * TILE; i++, p += 4) {
        arr[i] = (px[p] / 255 - MEAN[0]) / STD[0];
        arr[TILE * TILE + i] = (px[p + 1] / 255 - MEAN[1]) / STD[1];
        arr[2 * TILE * TILE + i] = (px[p + 2] / 255 - MEAN[2]) / STD[2];
      }
      const out = await model({ pixel_values: new Tensor('float32', arr, [1, 3, TILE, TILE]) });
      const hs = out.last_hidden_state;           // [1, 1+reg+G*G, D]
      const [, T, dim] = hs.dims;
      if (!F) { D = dim; F = new Float32Array(GW * GH * D); }
      const data = hs.data, off = T - G * G;
      for (let gy = 0; gy < G; gy++) {
        for (let gx = 0; gx < G; gx++) {
          const src = (off + gy * G + gx) * D;
          const dst = ((ty * G + gy) * GW + tx * G + gx) * D;
          let nrm = 0;
          for (let d = 0; d < D; d++) nrm += data[src + d] * data[src + d];
          nrm = 1 / Math.sqrt(nrm + 1e-8);
          for (let d = 0; d < D; d++) F[dst + d] = data[src + d] * nrm;
        }
      }
      post('stage', { text: `Analizez ${++done}/${n}…` });
    }
  }
  return { F, D, GW, GH, sx, sy };
}

// ------------------------------------------- CountingDINO: densitate din exemple
function densityCount(grid, exemplars) {
  const { F, D, GW, GH, sx, sy } = grid;
  const S = new Float32Array(GW * GH);
  const exCells = [];
  let maxArea = 1;

  for (const b of exemplars) {
    const x0 = Math.max(0, Math.floor((b[0] * sx) / PATCH)), y0 = Math.max(0, Math.floor((b[1] * sy) / PATCH));
    const x1 = Math.min(GW, Math.ceil((b[2] * sx) / PATCH)), y1 = Math.min(GH, Math.ceil((b[3] * sy) / PATCH));
    const kw = Math.max(1, x1 - x0), kh = Math.max(1, y1 - y0);
    exCells.push([x0, y0, x0 + kw, y0 + kh]);
    maxArea = Math.max(maxArea, kw * kh);
    // kernel = trăsăturile din chenar, ponderate eliptic (centrul contează mai mult)
    const K = [], cx = (kw - 1) / 2, cy = (kh - 1) / 2, rx = kw / 2 + 0.5, ry = kh / 2 + 0.5;
    let wsum = 0;
    for (let j = 0; j < kh; j++) for (let i = 0; i < kw; i++) {
      const w = Math.max(0.05, 1 - ((i - cx) / rx) ** 2 - ((j - cy) / ry) ** 2);
      K.push({ i: i - Math.floor(kw / 2), j: j - Math.floor(kh / 2), w, f: ((y0 + j) * GW + x0 + i) * D });
      wsum += w;
    }
    // corelație (template matching) pe toată grila
    for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
      let acc = 0, ws = 0;
      for (const k of K) {
        const xx = x + k.i, yy = y + k.j;
        if (xx < 0 || yy < 0 || xx >= GW || yy >= GH) continue;
        const p = (yy * GW + xx) * D;
        let dot = 0;
        for (let d = 0; d < D; d++) dot += F[p + d] * F[k.f + d];
        acc += k.w * dot; ws += k.w;
      }
      S[y * GW + x] += ws > 0 ? acc / ws / exemplars.length : 0;
    }
  }

  // min-max -> normalizare: fiecare exemplu trebuie să "valoreze" 1
  let mn = Infinity, mx = -Infinity;
  for (const v of S) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const Sm = S.map((v) => (v - mn) / (mx - mn + 1e-8));
  // masa unui obiect = suma răspunsului în jurul exemplului (fereastră 2x, după prag)
  const tau = 0.5;   // sub jumătate din vârful exemplelor = fundal
  let z = 0;
  for (const [x0, y0, x1, y1] of exCells) {
    const w = x1 - x0, h = y1 - y0;
    for (let y = Math.max(0, y0 - (w >> 1)); y < Math.min(GH, y1 + (h >> 1)); y++)
      for (let x = Math.max(0, x0 - (w >> 1)); x < Math.min(GW, x1 + (w >> 1)); x++) {
        const v = Sm[y * GW + x]; if (v >= tau) z += v;
      }
  }
  z /= exCells.length;
  let count = 0;
  const Dn = new Float32Array(GW * GH);
  for (let i = 0; i < Sm.length; i++) {
    if (Sm[i] >= tau) { Dn[i] = Sm[i] / (z + 1e-8); count += Dn[i]; }
  }

  // puncte (vârfuri locale) pentru afișare
  const avgK = Math.sqrt(exCells.reduce((a, c) => a + (c[2] - c[0]) * (c[3] - c[1]), 0) / exCells.length);
  const r = Math.max(1, Math.floor(avgK / 2));
  const peakMin = 0.6 * Math.max(...exCells.map(([x0, y0, x1, y1]) => {
    let m = 0; for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) m = Math.max(m, Sm[y * GW + x]); return m;
  }));
  const points = [];
  for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
    const v = Sm[y * GW + x];
    if (v < peakMin || Dn[y * GW + x] === 0) continue;
    let isMax = true;
    for (let j = -r; j <= r && isMax; j++) for (let i = -r; i <= r; i++) {
      const xx = x + i, yy = y + j;
      if ((i || j) && xx >= 0 && yy >= 0 && xx < GW && yy < GH && Sm[yy * GW + xx] > v) { isMax = false; break; }
    }
    if (isMax) points.push([((x + 0.5) * PATCH) / sx, ((y + 0.5) * PATCH) / sy, v]);
  }
  return { count, points };
}

// componente conexe (4-vecini) ale zonei proj > t
function components(proj, t, GW, GH) {
  const N = GW * GH, lab = new Int32Array(N).fill(-1), comps = [];
  for (let i = 0; i < N; i++) {
    if (proj[i] <= t || lab[i] >= 0) continue;
    const st = [i]; lab[i] = comps.length;
    let x0 = GW, y0 = GH, x1 = 0, y1 = 0, area = 0, cxs = 0, cys = 0;
    while (st.length) {
      const c = st.pop(), x = c % GW, y = (c / GW) | 0;
      area++; cxs += x; cys += y;
      if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y;
      if (x > 0 && proj[c - 1] > t && lab[c - 1] < 0) { lab[c - 1] = lab[i]; st.push(c - 1); }
      if (x < GW - 1 && proj[c + 1] > t && lab[c + 1] < 0) { lab[c + 1] = lab[i]; st.push(c + 1); }
      if (y > 0 && proj[c - GW] > t && lab[c - GW] < 0) { lab[c - GW] = lab[i]; st.push(c - GW); }
      if (y < GH - 1 && proj[c + GW] > t && lab[c + GW] < 0) { lab[c + GW] = lab[i]; st.push(c + GW); }
    }
    comps.push({ area, x0, y0, x1, y1, cx: cxs / area, cy: cys / area });
  }
  return comps;
}

// ------------------------------------------- automat: găsește singur obiectele
// 1) PC1 al trăsăturilor separă obiectele de fundal (fundalul = ce atinge marginile)
// 2) componente conexe = obiecte; 3) exemplu = obiectul de mărime tipică
function autoObjects(grid) {
  const { F, D, GW, GH, sx, sy } = grid;
  const N = GW * GH;
  const mean = new Float32Array(D);
  for (let i = 0; i < N; i++) for (let d = 0; d < D; d++) mean[d] += F[i * D + d] / N;
  // PC1 prin iterație de putere
  let v = new Float32Array(D).map((_, d) => Math.sin(d + 1));
  const proj = new Float32Array(N);
  for (let it = 0; it < 25; it++) {
    const nv = new Float32Array(D);
    for (let i = 0; i < N; i++) {
      let p = 0; for (let d = 0; d < D; d++) p += (F[i * D + d] - mean[d]) * v[d];
      for (let d = 0; d < D; d++) nv[d] += p * (F[i * D + d] - mean[d]);
    }
    let n = 0; for (let d = 0; d < D; d++) n += nv[d] * nv[d];
    n = Math.sqrt(n) || 1; v = nv.map((x) => x / n);
  }
  for (let i = 0; i < N; i++) { let p = 0; for (let d = 0; d < D; d++) p += (F[i * D + d] - mean[d]) * v[d]; proj[i] = p; }
  // semnul: marginile sunt fundal
  let edge = 0, ne = 0;
  for (let x = 0; x < GW; x++) { edge += proj[x] + proj[(GH - 1) * GW + x]; ne += 2; }
  for (let y = 0; y < GH; y++) { edge += proj[y * GW] + proj[y * GW + GW - 1]; ne += 2; }
  if (edge / ne > 0) for (let i = 0; i < N; i++) proj[i] = -proj[i];
  // pragul: îl caut pe cel care dă cele mai multe obiecte de mărime asemănătoare (ponderat cu mărimea)
  let mn = Infinity, mx = -Infinity;
  for (const p of proj) { if (p < mn) mn = p; if (p > mx) mx = p; }
  let bestScore = -1, comps = [];
  for (let k = 1; k <= 24; k++) {
    const t = mn + (k / 25) * (mx - mn);
    const cs = components(proj, t, GW, GH);
    if (!cs.length) continue;
    const big = Math.max(...cs.map((c) => c.area));
    const keep = cs.filter((c) => c.area >= Math.max(2, big * 0.08));
    const med = median(keep.map((c) => c.area));
    const good = keep.filter((c) => c.area >= med * 0.5 && c.area <= med * 2).length;
    const score = good * Math.sqrt(med);
    if (score > bestScore) { bestScore = score; comps = keep; }
  }
  // fără fragmente: sub 30% din mărimea tipică
  const medA = median(comps.map((c) => c.area));
  const objs = comps.filter((c) => c.area >= medA * 0.3);
  const toPx = (c) => [c.x0 * PATCH / sx, c.y0 * PATCH / sy, (c.x1 + 1) * PATCH / sx, (c.y1 + 1) * PATCH / sy];
  return { objs, toPx, toPt: (c) => [(c.cx + 0.5) * PATCH / sx, (c.cy + 0.5) * PATCH / sy, 1] };
}

// ---------------------------------------------------------------- API
async function count({ bitmap, W, H, boxes }) {
  const t0 = performance.now();
  const done = (r, mode) => ({ ...r, mode, seconds: (performance.now() - t0) / 1000 });
  let exemplars = boxes || [];

  if (!exemplars.length) {
    post('stage', { text: 'Caut obiectele…' });
    const g0 = await featureGrid(bitmap, W, H, [], 2);
    const { objs, toPx, toPt } = autoObjects(g0);
    if (!objs.length) return done({ count: 0, points: [], exemplars: [] }, 'automat');
    const med = median(objs.map((o) => o.area));
    const merged = objs.some((o) => o.area > med * 2.2);   // obiecte lipite -> trebuie densitate
    // puține obiecte, bine separate -> le numărăm direct
    if (objs.length <= 3 && !merged) {
      return done({ count: objs.length, points: objs.map(toPt), exemplars: objs.map(toPx) }, 'automat · obiecte separate');
    }
    exemplars = objs
      .filter((o) => o.area >= med * 0.6 && o.area <= med * 1.6)
      .sort((a, b) => Math.abs(a.area - med) - Math.abs(b.area - med))
      .slice(0, 3)
      .map(toPx);
    if (!exemplars.length) exemplars = [toPx(objs[0])];
  }

  post('stage', { text: 'Compar cu exemplul…' });
  const grid = await featureGrid(bitmap, W, H, exemplars);
  const r = densityCount(grid, exemplars);
  return done(
    { count: r.points.length, points: r.points, exemplars },
    (boxes?.length ? 'după exemplu' : 'automat') + ` · similaritate · densitate ≈ ${Math.round(r.count)}`,
  );
}

self.onmessage = async (e) => {
  const { id, cmd } = e.data;
  try {
    if (cmd === 'warmup') {
      await loadFeat();
      post('ready', { id, threads: env.backends.onnx.wasm.numThreads });
    } else if (cmd === 'count') {
      const res = await count(e.data);
      post('result', { id, res });
    }
  } catch (err) {
    console.error(err);
    post('error', { id, message: String(err?.message || err) });
  }
};
