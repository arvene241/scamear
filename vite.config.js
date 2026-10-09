// Cross-origin isolation unlocks multi-threaded WASM (SharedArrayBuffer), ~3x faster Whisper on phones.
const headers = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default {
  base: './',
  worker: { format: 'es' },
  server: { headers },
  preview: { headers },
};
