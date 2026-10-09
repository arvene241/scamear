// Network-first for the app shell so it loads in airplane mode. Model files are cached by Transformers.js / WebLLM themselves.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;

  // Share target (manifest share_target): keep the shared call recording, then open the app to analyse it.
  if (e.request.method === 'POST' && url.searchParams.has('share')) {
    e.respondWith((async () => {
      const file = (await e.request.formData()).get('audio');
      if (file) await (await caches.open('share')).put('shared-audio', new Response(file, { headers: { 'content-type': file.type || 'audio/mp4' } }));
      return Response.redirect(new URL('./?shared=1', self.registration.scope).href, 303);
    })());
    return;
  }

  if (e.request.method !== 'GET') return;
  e.respondWith(
    fetch(e.request)
      .then(res => { const copy = res.clone(); caches.open('shell').then(c => c.put(e.request, copy)); return res; })
      .catch(() => caches.match(e.request, { ignoreSearch: true, ignoreVary: true }))
  );
});
