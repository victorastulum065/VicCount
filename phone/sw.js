// VicCount service worker: tot din cache (merge în modul avion) +
// antete COOP/COEP ca să meargă WASM multi-thread (mai rapid).
const VERSION = 'viccount-v1';
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
      const res = await fetch(req);
      if (res.ok) await cache.put(req, res.clone());
      return isolate(res);
    } catch {
      return new Response('Offline și fișierul nu e în cache', { status: 503 });
    }
  })());
});
