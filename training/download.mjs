// Builds data.json from the BothBosu call dialogues, the UCI SMS Spam Collection and the Philippine spam SMS set.
// Fetch the raw CSV/zip files first (commands in README → Training).
import { readFileSync, writeFileSync } from 'node:fs';

function parseCsv(s) { // RFC4180: quoted fields may contain commas, quotes ("") and newlines
  const rows = []; let row = [], f = '', q = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === '"') { if (s[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && s[i + 1] === '\n') i++; row.push(f); rows.push(row); row = []; f = ''; }
    else f += c;
  }
  if (f || row.length) { row.push(f); rows.push(row); }
  const [head, ...body] = rows;
  return body.filter(r => r.length === head.length).map(r => Object.fromEntries(head.map((h, i) => [h, r[i]])));
}

// Speaker tags don't exist in our live transcripts, so drop them.
const clean = t => t.replace(/\b(caller|receiver|innocent|suspect)\s*:/gi, ' ').replace(/\s+/g, ' ').trim();

const data = [];
for (const [file, src] of [['scam-dialogue', 'scam-dialogue'], ['agent_conversation', 'multi-agent']]) {
  for (const split of ['train', 'test']) {
    const r = parseCsv(readFileSync(`csv/${file}_${split}.csv`, 'utf8'));
    for (const x of r) data.push({ src, kind: 'call', split, text: clean(x.dialogue), label: Number(x.label ?? x.labels), type: x.type });
    console.log(src, split, r.length);
  }
}
const sms = readFileSync('sms/SMSSpamCollection', 'utf8').trim().split(/\r?\n/).map(l => l.split('\t'));
// SMS set has no test split: hold out every 5th message.
sms.forEach(([lab, text], i) => data.push({ src: 'sms-spam', kind: 'sms', split: i % 5 === 0 ? 'test' : 'train', text: text.trim(), label: lab === 'spam' ? 1 : 0 }));
console.log('sms-spam', sms.length);

// Philippine spam/scam SMS (Kaggle bwandowando/philippine-spam-sms-messages, CC BY-NC-SA 4.0): spam only.
// Many messages are copies of one template with a different link, so split by template (first 40 letters)
// to keep near-duplicates out of the test set.
const templateKey = t => t.toLowerCase().replace(/[^a-z]/g, '').slice(0, 40);
const hash = s => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
const seen = new Set();
let ph = 0;
for (const r of parseCsv(readFileSync('ph-sms/SPAM_SMS.csv', 'utf8'))) {
  const text = r.text.replace(/<REAL NAME>/g, 'Juan').replace(/\s+/g, ' ').trim();
  const key = templateKey(text);
  if (text.length < 15 || /content not supported/i.test(text) || seen.has(text)) continue;
  seen.add(text);
  data.push({ src: 'ph-spam', kind: 'sms', split: hash(key) % 5 === 0 ? 'test' : 'train', text, label: 1 });
  ph++;
}
console.log('ph-spam', ph, '(after removing empty and exact duplicates)');

// Normal Filipino messages to balance it (training/ph-normal.mjs).
const { PH_NORMAL } = await import('./ph-normal.mjs');
PH_NORMAL.forEach((text, i) => data.push({ src: 'ph-normal', kind: 'sms', split: i % 5 === 2 ? 'test' : 'train', text, label: 0 }));
console.log('ph-normal', PH_NORMAL.length);

writeFileSync('data.json', JSON.stringify(data));
const count = f => data.filter(f).length;
console.log('total', data.length, '| scam/spam', count(d => d.label === 1), '| normal', count(d => d.label === 0));
console.log('call types:', [...new Set(data.filter(d => d.kind === 'call').map(d => d.type))].join(', '));
