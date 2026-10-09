// Trains a logistic-regression scam classifier on multilingual-e5-small embeddings.
// Output: classifier.json ({ w: number[384], b }) that the phone applies to the same embedding it already computes.
import { pipeline } from '@huggingface/transformers';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const data = JSON.parse(readFileSync('data.json', 'utf8'));

// The app scores the last 600 chars of the live transcript, so train on 600-char windows of each call.
const samples = [];
for (const d of data) {
  if (d.kind === 'sms') { samples.push({ ...d, text: d.text }); continue; }
  const t = d.text, mid = Math.max(0, Math.floor(t.length / 2) - 300);
  for (const win of new Set([t.slice(0, 600), t.slice(mid, mid + 600), t.slice(-600)])) samples.push({ ...d, text: win });
}

// Embedding cache keyed by text, so adding a dataset only embeds the new texts.
const cache = existsSync('emb.json') ? JSON.parse(readFileSync('emb.json', 'utf8')) : {};
const todo = [...new Set(samples.map(s => s.text).filter(t => !(t in cache)))];
if (todo.length) {
  const embed = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small', { dtype: 'q8' });
  const t0 = Date.now();
  for (let i = 0; i < todo.length; i += 64) {
    const batch = todo.slice(i, i + 64);
    const vecs = (await embed(batch.map(t => 'query: ' + t), { pooling: 'mean', normalize: true })).tolist();
    batch.forEach((t, j) => { cache[t] = vecs[j].map(x => +x.toFixed(5)); });
    if (i % 1280 === 0) console.log(`embedded ${i}/${todo.length} new texts (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
  writeFileSync('emb.json', JSON.stringify(cache));
}
const X = samples.map(s => cache[s.text]);

const EXCLUDE = (process.env.EXCLUDE || '').split(',').filter(Boolean); // e.g. EXCLUDE=ph-spam,ph-normal for the before/after comparison
// The Philippine set is spam-only, so its few normal messages are weighted up to balance that domain;
// otherwise the model learns "Tagalog + money talk = scam".
const NW = Number(process.env.NW || 10); // 1/5/10/20 tried: 10 stops family payment texts being flagged
const train = samples.map((s, i) => [X[i], s.label, s, s.src === 'ph-normal' ? NW : 1]).filter(r => r[2].split === 'train' && !EXCLUDE.includes(r[2].src));
// Add our own EN + Tagalog scam/normal scripts (src/patterns.js): the public sets lack family-emergency,
// investment, job and Tagalog scams, which are the most common in the Philippines. Weighted up since there are few.
const { PATTERNS } = await import('../src/patterns.js');
const PW = Number(process.env.PW || 10); // chosen on the held-out EN/TL set (10/30/60 tried)
{
  const embed = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small', { dtype: 'q8' });
  const pv = (await embed(PATTERNS.map(p => 'query: ' + p.text), { pooling: 'mean', normalize: true })).tolist();
  PATTERNS.forEach((p, i) => train.push([pv[i], p.type ? 1 : 0, { src: 'patterns' }, PW]));
}
const test = samples.map((s, i) => [X[i], s.label, s]).filter(r => r[2].split === 'test');
console.log(`train ${train.length} | test ${test.length}`);

// Logistic regression, full-batch gradient descent + L2, classes balanced by weight.
const D = X[0].length, sig = z => 1 / (1 + Math.exp(-z));
const w = new Float64Array(D); let b = 0;
const pos = train.filter(r => r[1]).reduce((s, r) => s + r[3], 0), all = train.reduce((s, r) => s + r[3], 0), neg = all - pos;
const cw = [all / (2 * neg), all / (2 * pos)];
const predict = x => sig(x.reduce((s, v, i) => s + v * w[i], b));
for (let epoch = 0; epoch < 600; epoch++) {
  const gw = new Float64Array(D); let gb = 0;
  for (const [x, y, , sw] of train) {
    const e = (predict(x) - y) * cw[y] * sw;
    for (let i = 0; i < D; i++) gw[i] += e * x[i];
    gb += e;
  }
  for (let i = 0; i < D; i++) w[i] -= 2.0 * (gw[i] / all + 1e-4 * w[i]);
  b -= 2.0 * gb / all;
}

function report(name, rows) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const [x, y] of rows) { const p = predict(x) >= 0.5; if (p && y) tp++; else if (p) fp++; else if (y) fn++; else tn++; }
  const acc = (tp + tn) / rows.length, prec = tp / (tp + fp || 1), rec = tp / (tp + fn || 1);
  console.log(`${name.padEnd(22)} n=${String(rows.length).padStart(5)}  accuracy ${(acc * 100).toFixed(1)}%  precision ${(prec * 100).toFixed(1)}%  recall ${(rec * 100).toFixed(1)}%  (false alarms ${fp}, missed ${fn})`);
}
report('TEST all', test);
for (const src of ['scam-dialogue', 'multi-agent', 'sms-spam', 'ph-spam', 'ph-normal']) report(`TEST ${src}`, test.filter(r => r[2].src === src));

writeFileSync('../src/classifier.json', JSON.stringify({
  model: 'Xenova/multilingual-e5-small',
  trainedOn: 'BothBosu/scam-dialogue + BothBosu/multi-agent-scam-conversation (Apache-2.0) + UCI SMS Spam Collection (CC BY 4.0) + Philippine Spam SMS by bwandowando on Kaggle (CC BY-NC-SA 4.0) + ScamEar EN/TL scripts and normal PH messages',
  w: [...w].map(v => +v.toFixed(5)), b: +b.toFixed(5),
}));
console.log('wrote src/classifier.json');
