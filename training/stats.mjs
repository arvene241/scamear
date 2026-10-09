// Full evaluation report on held-out data, scored exactly like the app:
// risk = max(rules, scam memory, classifier), capped at 30 for a legit code notice; "scam" = risk ≥ 35.
// Run after download.mjs + train.mjs (uses data.json + emb.json).
import { pipeline } from '@huggingface/transformers';
import { readFileSync } from 'node:fs';
import { PATTERNS, matchPattern } from '../src/patterns.js';
import { scan, isCodeNotice } from '../src/rules.js';
import { FIL_TEST } from './fil-test.mjs';

const data = JSON.parse(readFileSync('data.json', 'utf8'));
const cache = JSON.parse(readFileSync('emb.json', 'utf8'));
const { w, b } = JSON.parse(readFileSync('../src/classifier.json', 'utf8'));
const samples = []; // same 600-char windows as train.mjs
for (const d of data) {
  if (d.kind === 'sms') { samples.push(d); continue; }
  const t = d.text, mid = Math.max(0, Math.floor(t.length / 2) - 300);
  for (const win of new Set([t.slice(0, 600), t.slice(mid, mid + 600), t.slice(-600)])) samples.push({ ...d, text: win });
}
const embed = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small', { dtype: 'q8' });
const vecs = async t => (await embed(t.map(x => 'query: ' + x), { pooling: 'mean', normalize: true })).tolist();
const pv = await vecs(PATTERNS.map(p => p.text));
const prob = v => 1 / (1 + Math.exp(-v.reduce((s, x, i) => s + x * w[i], b)));

function score(text, x) {
  const p = prob(x), clf = Math.max(0, Math.min(1, (p - 0.5) / 0.3)) * 100, mem = matchPattern(x, pv).risk;
  const notice = isCodeNotice(text), rules = notice ? 0 : scan(text).score;
  return { p, clf, mem, rules, app: Math.min(notice ? 30 : 100, Math.max(rules, mem, clf)) };
}
const test = samples.filter(s => s.split === 'test').map(s => ({ s, ...score(s.text, cache[s.text]) }));

const pct = x => (x * 100).toFixed(1) + '%';
function metrics(rows, key) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const r of rows) { const p = r[key] >= 35; if (p && r.s.label) tp++; else if (p) fp++; else if (r.s.label) fn++; else tn++; }
  const prec = tp / (tp + fp || 1), rec = tp / (tp + fn || 1);
  return { n: rows.length, tp, fp, tn, fn, acc: (tp + tn) / rows.length, prec, rec, f1: 2 * prec * rec / (prec + rec || 1), fpr: fp / (fp + tn || 1) };
}
function auc(rows) { // probability a random scam scores above a random normal message
  const pos = rows.filter(r => r.s.label).map(r => r.p), neg = rows.filter(r => !r.s.label).map(r => r.p);
  let s = 0; for (const a of pos) for (const c of neg) s += a > c ? 1 : a === c ? 0.5 : 0;
  return s / (pos.length * neg.length);
}

console.log(`== Held-out test set (n=${test.length}, never seen in training) ==`);
for (const [name, key] of [['Rules only', 'rules'], ['Scam memory only', 'mem'], ['Trained classifier only', 'clf'], ['SHIPPED APP (all combined)', 'app']]) {
  const m = metrics(test, key);
  console.log(`${name.padEnd(27)} acc ${pct(m.acc)}  precision ${pct(m.prec)}  recall ${pct(m.rec)}  F1 ${m.f1.toFixed(3)}  false-alarm rate ${pct(m.fpr)}`);
}
const m = metrics(test, 'app');
console.log(`\nConfusion matrix (shipped app):\n                predicted SCAM  predicted OK\n  actual SCAM   ${String(m.tp).padStart(14)}  ${String(m.fn).padStart(12)}\n  actual OK     ${String(m.fp).padStart(14)}  ${String(m.tn).padStart(12)}`);
console.log(`\nClassifier ROC-AUC: ${auc(test).toFixed(3)}`);

console.log('\n== By source (shipped app) ==');
for (const src of ['scam-dialogue', 'multi-agent', 'sms-spam', 'ph-spam', 'ph-normal']) {
  const r = metrics(test.filter(t => t.s.src === src), 'app');
  console.log(`${src.padEnd(14)} n=${String(r.n).padStart(4)}  acc ${pct(r.acc)}  ${r.tp + r.fn ? `recall ${pct(r.rec)}` : `false alarms ${r.fp}/${r.n}`}`);
}
console.log('\n== By call type (shipped app; % flagged as scam) ==');
const calls = test.filter(t => t.s.kind === 'call');
for (const type of [...new Set(calls.map(t => t.s.type))].sort()) {
  const rows = calls.filter(t => t.s.type === type), flagged = rows.filter(t => t.app >= 35).length;
  console.log(`${type.padEnd(14)} ${rows.filter(t => t.s.label).length > rows.length / 2 ? 'SCAM  ' : 'NORMAL'} n=${String(rows.length).padStart(3)}  flagged ${pct(flagged / rows.length)}`);
}

console.log('\n== Hand-written EN + Tagalog/Taglish set (training/fil-test.mjs) ==');
const fv = await vecs(FIL_TEST.map(t => t[1]));
const fil = FIL_TEST.map(([label, text], i) => ({ s: { label }, text, ...score(text, fv[i]) }));
for (const key of ['mem', 'clf', 'app']) console.log(`${key.padEnd(4)} ${fil.filter(r => (r[key] >= 35) === !!r.s.label).length}/${fil.length}`);
fil.filter(r => (r.app >= 35) !== !!r.s.label).forEach(r => console.log(`  wrong: ${r.s.label ? 'scam' : 'normal'} app=${Math.round(r.app)} | ${r.text.slice(0, 70)}`));
