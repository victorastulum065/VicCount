// VicCount — motorul de numărare, rulează în Web Worker, 100% pe dispozitiv.
//  1) Grounding DINO tiny  -> găsește obiectele după text ("object" în modul automat)
//  2) DINOv2-reg small     -> numărare după exemplu (metoda CountingDINO, WACV 2026):
//                             similaritate de trăsături cu exemplul -> hartă de densitate -> sumă
import {
  pipeline, AutoModel, Tensor, RawImage, env,
} from './vendor/transformers.min.js';

const BASE = new URL('./', self.location.href).href;
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = BASE + 'models/';
env.useBrowserCache = false;                       // cache-ul îl face service worker-ul
env.backends.onnx.wasm.wasmPaths = BASE + 'vendor/';
env.backends.onnx.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const DET_ID = 'grounding-dino-tiny';
const FEAT_ID = 'dinov2-reg-small';
const TILE = 518, PATCH = 14, G = TILE / PATCH;    // 37x37 patch-uri / bucată
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

let detector = null, featModel = null;

const post = (type, data) => self.postMessage({ type, ...data });
const progress = (label) => (p) => {
  if (p.status === 'progress' && p.total) post('progress', { label, file: p.file, loaded: p.loaded, total: p.total });
};

function loadDetector() {
  if (!detector) {
    detector = pipeline('zero-shot-object-detection', DET_ID, {
      dtype: 'q8', device: 'wasm', progress_callback: progress('Detector'),
    });
  }
  return detector;
}
function loadFeat() {
  if (!featModel) {
    featModel = AutoModel.from_pretrained(FEAT_ID, {
      dtype: 'q8', device: 'wasm', progress_callback: progress('Trăsături'),
    });
  }
  return featModel;
}

// ---------------------------------------------------------------- utilitare
function iou(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}
function nms(dets, thr = 0.5) {
  const out = [];
  for (const d of dets.sort((a, b) => b.score - a.score)) {
    if (out.every((o) => iou(o.box, d.box) < thr)) out.push(d);
  }
  return out;
}
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };

// ---------------------------------------------------------------- detector
async function detect(bitmap, label, W, H) {
  const det = await loadDetector();
  const c = new OffscreenCanvas(W, H);
  c.getContext('2d').drawImage(bitmap, 0, 0, W, H);
  const id = c.getContext('2d').getImageData(0, 0, W, H);
  const img = new RawImage(id.data, W, H, 4).rgb();
  const q = label.trim().toLowerCase().replace(/\.+$/, '') + '.';
  const res = await det(img, [q], { threshold: 0.1 });
  const dets = res
    .map((r) => ({ score: r.score, box: [r.box.xmin, r.box.ymin, r.box.xmax, r.box.ymax] }))
    .filter((d) => {
      const a = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]);
      return a > 4 && a < 0.97 * W * H;          // doar zgomot / chenarul întregii poze
    });
  const kept = nms(dets, 0.5);
  // un chenar uriaș care conține mai multe obiecte găsite = grupul lor, nu un obiect
  return kept.filter((d) => {
    const a = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]);
    if (a < 0.3 * W * H) return true;
    const inside = kept.filter((o) => o !== d && o.box[0] >= d.box[0] - 2 && o.box[1] >= d.box[1] - 2
      && o.box[2] <= d.box[2] + 2 && o.box[3] <= d.box[3] + 2).length;
    return inside < 2;
  });
}

// prag: 0.25 normal; dacă nimic nu trece, coboară după cel mai bun scor (obiect unic, fotografiat de aproape)
function detThreshold(dets) {
  const best = dets.length ? dets[0].score : 0;
  return best >= 0.25 ? 0.25 : Math.max(0.12, best * 0.75);
}

function pickPseudoExemplars(dets, W, H, k = 3) {
  const good = dets.filter((d) => d.score >= detThreshold(dets));
  if (!good.length) return [];
  const areas = good.map((d) => (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]));
  const med = median(areas);
  return good
    .filter((d, i) => areas[i] < med * 2.5 && areas[i] > med / 2.5 && areas[i] < 0.5 * W * H)
    .slice(0, k)
    .map((d) => d.box);
}

// ------------------------------------------------- trăsături DINOv2 pe bucăți
async function featureGrid(bitmap, W, H, exemplars) {
  const model = await loadFeat();
  // scară: exemplul să ocupe ~4 patch-uri (56px) -> rezoluție bună și pt. obiecte mici
  const exSize = median(exemplars.map((b) => Math.sqrt((b[2] - b[0]) * (b[3] - b[1]))));
  let s = Math.max(TILE / Math.max(W, H), 56 / Math.max(exSize, 1));
  let cols = Math.ceil((W * s) / TILE), rows = Math.ceil((H * s) / TILE);
  while (cols * rows > 16) { s *= 0.85; cols = Math.ceil((W * s) / TILE); rows = Math.ceil((H * s) / TILE); }
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

// ---------------------------------------------------------------- API
async function count({ bitmap, W, H, text, boxes }) {
  const t0 = performance.now();
  let mode, dets = [], exemplars = boxes || [];

  if (!exemplars.length) {
    post('stage', { text: 'Caut obiectele…' });
    dets = await detect(bitmap, text || 'object', W, H);
    exemplars = pickPseudoExemplars(dets, W, H);
    mode = text ? `text „${text}”` : 'automat';
  } else {
    mode = 'după exemplu';
  }

  dets.sort((a, b) => b.score - a.score);
  const thr = detThreshold(dets);
  const detCount = dets.filter((d) => d.score >= thr).length;
  // puține obiecte și avem text -> detectorul e cel mai precis; altfel densitate
  if (!boxes?.length && (detCount < 25 || !exemplars.length)) {
    return {
      count: detCount,
      points: dets.filter((d) => d.score >= thr).map((d) => [(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2, d.score]),
      boxes: dets, exemplars, mode: mode + ' · detector', seconds: (performance.now() - t0) / 1000,
    };
  }

  post('stage', { text: 'Compar cu exemplul…' });
  const grid = await featureGrid(bitmap, W, H, exemplars);
  const r = densityCount(grid, exemplars);
  return {
    count: r.points.length,            // = punctele afișate (densitatea doar pt. info)
    points: r.points,
    boxes: dets, exemplars,
    mode: mode + ' · similaritate' + (detCount ? ` (detector: ${detCount})` : '') + ` · densitate ≈ ${Math.round(r.count)}`,
    seconds: (performance.now() - t0) / 1000,
  };
}

self.onmessage = async (e) => {
  const { id, cmd } = e.data;
  try {
    if (cmd === 'warmup') {
      await loadDetector();
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
