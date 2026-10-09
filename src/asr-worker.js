import { pipeline } from '@huggingface/transformers';

let asr;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      asr = await pipeline('automatic-speech-recognition', data.model, {
        device: 'wasm',
        dtype: 'q8',
        progress_callback: p => p.status === 'progress' && self.postMessage({ type: 'progress', file: p.file, progress: p.progress }),
      });
      self.postMessage({ type: 'ready' });
    } else if (data.type === 'transcribe') {
      const t0 = performance.now();
      // Tagalog source + translate: Tagalog, Taglish and English calls all come out as English,
      // so the English rules and scam memory work on them.
      const { text } = await asr(data.audio, { language: 'tagalog', task: 'translate' });
      console.debug(`ASR ${Math.round(performance.now() - t0)}ms for ${(data.audio.length / 16000).toFixed(1)}s:`, text);
      self.postMessage({ type: 'text', run: data.run, text });
    }
  } catch (e) {
    self.postMessage({ type: 'error', message: String(e?.message || e) });
  }
};
