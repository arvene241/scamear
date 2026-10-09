// Sentence embeddings on the CPU (WASM), scored two ways: nearest known scam script (names the scam type)
// and a classifier trained on 8.7k labeled calls/SMS (training/train.mjs). Works on any phone.
import { pipeline } from '@huggingface/transformers';
import { PATTERNS, matchPattern } from './patterns.js';
import clf from './classifier.json';

let embed, patternVecs;
const vecs = async texts => (await embed(texts.map(t => 'query: ' + t), { pooling: 'mean', normalize: true })).tolist(); // e5 expects the prefix

// p = 0.5 → 0 risk, p = 0.8 → 100.
const learnedRisk = v => {
  const p = 1 / (1 + Math.exp(-v.reduce((s, x, i) => s + x * clf.w[i], clf.b)));
  return Math.round(Math.max(0, Math.min(1, (p - 0.5) / 0.3)) * 100);
};

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      embed = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small', { // understands Tagalog + English
        device: 'wasm',
        dtype: 'q8',
        progress_callback: p => p.status === 'progress' && self.postMessage({ type: 'progress', file: p.file, progress: p.progress }),
      });
      patternVecs = await vecs(PATTERNS.map(p => p.text));
      self.postMessage({ type: 'ready' });
    } else if (data.type === 'match') {
      const [vec] = await vecs([data.text]);
      const m = matchPattern(vec, patternVecs), learned = learnedRisk(vec);
      // Max of the two: each catches scams the other misses (see README → Evaluation).
      self.postMessage({ type: 'match', run: data.run, match: { ...m, memoryRisk: m.risk, learned, risk: Math.max(m.risk, learned) } });
    }
  } catch (e) {
    self.postMessage({ type: 'error', message: String(e?.message || e) });
  }
};
