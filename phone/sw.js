// VicCount service worker: tot din cache (merge în modul avion) +
// antete COOP/COEP ca să meargă WASM multi-thread (mai rapid).
const VERSION = 'viccount-v2';
const SHELL = ['./', './index.html', './app.js', './worker.js', './manifest.webmanifest', './icon-192.png', './icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== VERSION) await caches.delete(k);
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
    const cache = await caches.open(VERSION);
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit) {
      // app shell: actualizează în fundal; modelele rămân cum sunt
      if (!req.url.includes('/models/') && !req.url.includes('/vendor/')) {
        fetch(req).then((r) => r.ok && cache.put(req, r)).catch(() => {});
      }
      return isolate(hit);
    }
    try {
      let res = await fetch(req);
      // fișierele mari (>90 MB) sunt publicate în bucăți: <fișier>.parts.json + .part0, .part1…
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
