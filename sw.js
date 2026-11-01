// VicCount service worker: merge offline (modul avion) +
// antete COOP/COEP ca să meargă WASM multi-thread (mai rapid).
const MODELS = 'viccount-v2';          // modele + motor (rămân la actualizări)
const VERSION = 'viccount-shell-4';    // aplicația
const SHELL = ['./', './index.html', './app.js', './worker.js', './manifest.webmanifest', './icon-192.png', './icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== VERSION && k !== MODELS) await caches.delete(k);
    // detectorul vechi (~200 MB) nu mai e folosit -> eliberează spațiul
    const mc = await caches.open(MODELS);
    for (const r of await mc.keys()) if (r.url.includes('/grounding-dino-tiny/')) await mc.delete(r);
    await self.clients.claim();
  })());
});

function isolate(res) {
  if (!res || res.status === 0 || res.type === 'opaque') return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

// fișierele mari (>90 MB) sunt publicate în bucăți: <fișier>.parts.json + .part0, .part1…
async function joinParts(url) {
  const meta = await fetch(url + '.parts.json');
  if (!meta.ok) return new Response('Not found', { status: 404 });
  const { parts, size } = await meta.json();
  let i = 0, reader = null;
  const body = new ReadableStream({
    async pull(ctrl) {
      for (;;) {
        if (!reader) {
          if (i >= parts) { ctrl.close(); return; }
          const r = await fetch(`${url}.part${i++}`);
          if (!r.ok) { ctrl.error(new Error('part ' + (i - 1) + ' ' + r.status)); return; }
          reader = r.body.getReader();
        }
        const { done, value } = await reader.read();
        if (done) { reader = null; continue; }
        ctrl.enqueue(value);
        return;
      }
    },
  });
  return new Response(body, {
    status: 200,
    headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(size) },
  });
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith((async () => {
    const heavy = req.url.includes('/models/') || req.url.includes('/vendor/');
    if (!heavy) {
      // aplicația: întâi rețeaua (actualizări imediate), altfel din cache (offline)
      const cache = await caches.open(VERSION);
      try {
        const res = await fetch(req);
        if (res.ok) cache.put(req, res.clone()).catch(() => {});
        return isolate(res);
      } catch {
        const hit = await caches.match(req, { ignoreSearch: true });
        return hit ? isolate(hit) : new Response('Offline', { status: 503 });
      }
    }
    // modele + motor: din cache; prima dată din rețea, apoi salvate
    const cache = await caches.open(MODELS);
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit) return isolate(hit);
    try {
      let res = await fetch(req);
      if (res.status === 404 && req.url.endsWith('.onnx')) res = await joinParts(req.url);
      if (res.ok) {
        const [a, b] = res.body.tee();
        const headers = new Headers(res.headers);
        cache.put(req, new Response(a, { status: 200, headers })).catch(() => {});
        return isolate(new Response(b, { status: 200, headers }));
      }
      return isolate(res);
    } catch {
      return new Response('Offline și fișierul nu e în cache', { status: 503 });
    }
  })());
});
