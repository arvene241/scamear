import { CreateMLCEngine } from '@mlc-ai/web-llm';
import { scan, isCodeNotice } from './rules.js';
import asrWorkerUrl from './asr-worker.js?worker&url';
import embedWorkerUrl from './embed-worker.js?worker&url';

const params = new URLSearchParams(location.search);
// Phones get whisper-base in low-memory mode: whisper-small grew a Galaxy S23 FE Chrome tab past 3.7 GB and
// Android killed it, while base checks a 35 s call in ~31 s. Computers keep the more accurate whisper-small.
const PHONE = /Android|iPhone|iPad/i.test(navigator.userAgent);
const ASR_MODEL = params.get('asr') || (PHONE ? 'Xenova/whisper-base' : 'Xenova/whisper-small'); // multilingual: hears Tagalog/Taglish
const LLM_SIZE = params.get('llm') || '1.5B';
// Tuning knobs for low-memory phones: ?threads=N caps WASM threads, ?enc=q4 uses the smaller Whisper encoder.
const THREADS = Number(params.get('threads')) || undefined;
const LOWMEM = params.has('lowmem') ? params.get('lowmem') === '1' : PHONE; // no ONNX memory arena
const ASR_DTYPE = params.get('enc') === 'q4' ? { encoder_model: 'q4', decoder_model_merged: 'q8' } : 'q8';
const SAMPLE_RATE = 16000;

const $ = id => document.getElementById(id);
// run increases on every Clear, so late answers from the models for an earlier call or text are ignored.
const state = { run: 0, transcript: '', pattern: null, llm: null, llmBusy: false, llmPending: false, ai: null, seenFlags: new Set() };

// ---------- Model loading ----------
const ASR_URL = new URL(asrWorkerUrl, location.href);
const asr = new Worker(ASR_URL, { type: 'module' });
let asrReady = false, asrBusy = false;
const asrProgress = {};

asr.onmessage = ({ data }) => {
  if (data.type === 'progress') {
    setStatus('ears', `Downloading… ${avgProgress(asrProgress, data)}%`);
  } else if (data.type === 'ready') {
    asrReady = true;
    setStatus('ears', 'Ready ✓', true);
    try { localStorage.setItem('scamear-models', '1'); } catch {}
    updateReady();
    nextFileChunk();
  } else if (data.type === 'text') {
    asrBusy = false;
    const text = cleanAsr(data.text);
    if (text && data.run === state.run) addTranscript(text);
    nextFileChunk();
  } else if (data.type === 'error') {
    asrBusy = false;
    setStatus('ears', 'Error: ' + data.message);
    nextFileChunk();
  }
};

// Scam memory + trained classifier: multilingual-e5 embeddings on the CPU, so every phone gets AI judgment.
const EMBED_URL = new URL(embedWorkerUrl, location.href);
const embedder = new Worker(EMBED_URL, { type: 'module' });
const embedProgress = {};
let embedReady = false, embedBusy = false, embedPending = false;

embedder.onmessage = ({ data }) => {
  if (data.type === 'progress') {
    setStatus('memory', `Downloading… ${avgProgress(embedProgress, data)}%`);
  } else if (data.type === 'ready') {
    embedReady = true;
    setStatus('memory', 'Ready ✓', true);
    if (state.transcript) matchPatterns();
  } else if (data.type === 'match') {
    embedBusy = false;
    if (data.run === state.run) state.pattern = data.match;
    render();
    if (embedPending) { embedPending = false; matchPatterns(); }
  } else if (data.type === 'error') {
    embedBusy = false;
    setStatus('memory', 'Error: ' + data.message);
  }
};

function matchPatterns() {
  if (!embedReady) return;
  if (embedBusy) { embedPending = true; return; }
  embedBusy = true;
  embedder.postMessage({ type: 'match', run: state.run, text: state.transcript.slice(-600) }); // recent speech = current topic
}

// Bonus reasoning LLM, only where Chrome exposes the GPU via WebGPU (it doesn't on e.g. Exynos/Xclipse phones).
async function loadLlm() {
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  if (!adapter) return setStatus('brain', 'Skipped: no WebGPU on this phone');
  const quant = adapter.features.has('shader-f16') ? 'q4f16_1' : 'q4f32_1';
  try {
    const engine = await CreateMLCEngine(`Qwen2.5-${LLM_SIZE}-Instruct-${quant}-MLC`, {
      initProgressCallback: p => setStatus('brain', `Loading on GPU… ${Math.round(p.progress * 100)}%`),
    });
    state.llm = async messages => (await engine.chat.completions.create({
      messages, temperature: 0,
      max_tokens: 120, // no response_format: WebLLM 0.2.85's json_object grammar throws GrammarMatcherInitError
    })).choices[0].message.content;
    setStatus('brain', `Ready ✓ (${LLM_SIZE} · GPU)`, true);
    if (state.transcript) askLlm();
  } catch (e) {
    setStatus('brain', 'Skipped: ' + (e.message || e));
  }
}

let modelsStarted = false;
function startModels() {
  if (modelsStarted) return;
  modelsStarted = true;
  $('load').disabled = true;
  asr.postMessage({ type: 'load', model: ASR_MODEL, dtype: ASR_DTYPE, threads: THREADS, lowmem: LOWMEM });
  embedder.postMessage({ type: 'load', threads: THREADS, lowmem: LOWMEM });
  loadLlm();
}
$('load').onclick = startModels;

function avgProgress(map, { file, progress }) {
  map[file] = progress;
  const vals = Object.values(map);
  return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
}

function setStatus(which, text, ok = false) {
  $(which).textContent = text;
  $(which).classList.toggle('ok', ok);
}

function updateReady() {
  if (asrReady) {
    $('setup').hidden = true;
    $('main').hidden = false;
  }
}

// ---------- Microphone → Whisper ----------
let audioCtx, stream, wakeLock, buffered = [], bufferedLen = 0, pumpTimer;

async function startListening() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: true, autoGainControl: true } });
  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  const src = audioCtx.createMediaStreamSource(stream);
  // ponytail: ScriptProcessor is deprecated but universally supported; move to AudioWorklet if it ever glitches.
  const proc = audioCtx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = e => {
    const chunk = new Float32Array(e.inputBuffer.getChannelData(0));
    buffered.push(chunk);
    bufferedLen += chunk.length;
  };
  src.connect(proc);
  proc.connect(audioCtx.destination);
  pumpTimer = setInterval(pump, 250);
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {}
  $('listen').textContent = '■ Stop listening';
  $('listen').classList.add('live');
}

function stopListening() {
  clearInterval(pumpTimer);
  stream?.getTracks().forEach(t => t.stop());
  audioCtx?.close();
  wakeLock?.release();
  buffered = []; bufferedLen = 0;
  $('listen').textContent = '● Start listening';
  $('listen').classList.remove('live');
}

// Send ≥5s of audio to Whisper whenever it is idle (longer chunks = better Tagalog→English context).
function pump() {
  if (asrBusy || bufferedLen < SAMPLE_RATE * 5) return;
  const audio = new Float32Array(bufferedLen);
  let off = 0;
  for (const c of buffered) { audio.set(c, off); off += c.length; }
  buffered = []; bufferedLen = 0;
  if (isSilent(audio)) return;
  asrBusy = true;
  asr.postMessage({ type: 'transcribe', run: state.run, audio }, [audio.buffer]);
}

// Whisper hallucinates phrases like "Thank you." on silence, so silent audio is never sent.
const isSilent = audio => Math.sqrt(audio.reduce((s, x) => s + x * x, 0) / audio.length) < 0.005;

const HALLUCINATIONS = /^(thank you\.?|thanks for watching!?|you|\.+)$/i;
function cleanAsr(t) {
  t = t.replace(/\[[^\]]*\]|\([^)]*\)|\*[^*]*\*/g, '').trim();
  return t.length < 2 || HALLUCINATIONS.test(t) ? '' : t;
}

$('listen').onclick = () => (stream?.active ? stopListening() : startListening().catch(e => alertBar('Mic error: ' + e.message)));

// ---------- Call recording → Whisper (after the call) ----------
// Android doesn't let third-party apps capture call audio live, but the phone's own recorder
// (e.g. Samsung "Auto record calls") saves .m4a files that can be checked here or shared to ScamEar.
const WINDOW = SAMPLE_RATE * 30; // Whisper's input window: one call per 30 s is the cheapest way through
let fileChunks = [], fileTotal = 0, fileSeconds = 0;

async function analyseRecording(blob) {
  if (stream?.active) stopListening();
  reset();
  $('rec').hidden = false;
  setRec(0, 'Reading the recording…');
  let audio;
  try {
    // An OfflineAudioContext at 16 kHz decodes m4a/mp3/wav and resamples to what Whisper expects.
    audio = await new OfflineAudioContext(1, 1, SAMPLE_RATE).decodeAudioData(await blob.arrayBuffer());
  } catch {
    return setRec(0, "Couldn't read this file. Choose a call recording such as .m4a, .mp3 or .wav.");
  }
  const mono = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const ch = audio.getChannelData(c);
    for (let i = 0; i < ch.length; i++) mono[i] += ch[i] / audio.numberOfChannels;
  }
  fileChunks = [];
  for (let i = 0; i < mono.length; i += WINDOW) fileChunks.push(mono.slice(i, i + WINDOW));
  fileTotal = fileChunks.length;
  fileSeconds = Math.round(mono.length / SAMPLE_RATE);
  if (!asrReady) setRec(0, 'Loading the speech model first…');
  nextFileChunk();
}

function nextFileChunk() {
  if (!asrReady || asrBusy || !fileTotal) return;
  while (fileChunks.length && isSilent(fileChunks[0])) fileChunks.shift();
  const done = fileTotal - fileChunks.length;
  if (!fileChunks.length) {
    setRec(1, `Checked the whole call (${Math.floor(fileSeconds / 60)}:${String(fileSeconds % 60).padStart(2, '0')}).`);
    fileTotal = 0;
    return;
  }
  setRec(done / fileTotal, `Listening to the recording… part ${done + 1} of ${fileTotal}`);
  const audio = fileChunks.shift();
  asrBusy = true;
  asr.postMessage({ type: 'transcribe', run: state.run, audio }, [audio.buffer]);
}

function setRec(fraction, text) {
  $('rec-progress').value = fraction;
  $('rec-status').textContent = text;
}

$('pick').onclick = () => $('recording').click();
$('recording').onchange = () => {
  const file = $('recording').files[0];
  $('recording').value = '';
  if (file) analyseRecording(file);
};

// ---------- Analysis ----------
function addTranscript(text) {
  state.transcript += (state.transcript ? ' ' : '') + text;
  render();
  matchPatterns();
  askLlm();
}

async function askLlm() {
  if (!state.llm) return;
  if (state.llmBusy) { state.llmPending = true; return; }
  state.llmBusy = true;
  const run = state.run;
  $('ai').classList.add('thinking');
  try {
    const reply = await state.llm([
      { role: 'system', content:
        'You are ScamEar, a scam detector running on the user\'s phone. You get a transcript of what a caller said or a message ' +
        'the user received (speech recognition may contain errors), plus red flags found by keyword rules. ' +
        'Judge whether it is a scam attempt. Banks, police and companies never ask for OTPs, PINs, gift cards or remote access. ' +
        'Ordinary conversation is low risk. Reply with two lines, like these examples:\n' +
        'RISK: 95\nREASON: The caller demands an OTP and threatens to freeze the account.\n\n' +
        'RISK: 5\nREASON: A friendly chat about lunch plans with no requests for money or codes.' },
      { role: 'user', content: `Red flags: ${scan(state.transcript).flags.map(f => f.label).join('; ') || 'none'}\n\nTranscript: ${state.transcript.slice(-1200)}` },
    ]);
    console.debug('LLM reply:', reply);
    if (run === state.run) state.ai = parseVerdict(reply);
  } catch (e) {
    console.warn('LLM failed', e);
  }
  state.llmBusy = false;
  $('ai').classList.remove('thinking');
  render();
  if (state.llmPending) { state.llmPending = false; askLlm(); }
}

// Small models drift from the format (JSON, markdown, extra prose), so pull the two fields out loosely.
function parseVerdict(reply) {
  const risk = Number(reply.match(/risk\W{0,4}(\d{1,3})/i)?.[1] ?? reply.match(/\b(\d{1,3})\b/)?.[1]);
  if (!Number.isFinite(risk)) throw new Error('No risk in LLM reply: ' + reply);
  const reason = (reply.match(/reason\W{0,4}([^\n"}]+)/i)?.[1] ?? reply.replace(/risk\W{0,4}\d+/i, '')).trim();
  return { risk: Math.min(100, risk), reason: reason.slice(0, 240) };
}
const ADVICE = {
  scam: 'Hang up now. Never share codes or send money. Call the company back using the number on their official website or your card.',
  warn: "Slow down. Don't share codes, passwords or money. Verify by calling the official number yourself.",
  safe: 'No scam tactics detected so far.',
};

let lastFlagsHtml = '';
function render() {
  const notice = isCodeNotice(state.transcript); // a bank/GCash code notice is normal; being asked for the code is not
  const { flags, score } = notice ? { flags: [], score: 0 } : scan(state.transcript);
  const risk = Math.min(notice ? 30 : 100, Math.max(score, state.pattern?.risk ?? 0, state.ai?.risk ?? 0));
  const level = risk >= 70 ? 'scam' : risk >= 35 ? 'warn' : 'safe';

  document.body.dataset.level = state.transcript ? level : '';
  $('gauge').style.setProperty('--risk', risk);
  $('score').textContent = risk;
  $('verdict').textContent = !state.transcript ? 'Listening for red flags…'
    : level === 'scam' ? 'LIKELY SCAM — HANG UP' : level === 'warn' ? 'Suspicious — be careful' : 'Sounds normal';

  const fresh = flags.filter(f => !state.seenFlags.has(f.id));
  if (fresh.length) {
    fresh.forEach(f => state.seenFlags.add(f.id));
    navigator.vibrate?.(level === 'scam' ? [300, 100, 300] : 150);
  }
  const flagsHtml = flags.map(f => `<li><b>🚩 ${f.label}</b><span>${f.tip}</span></li>`).join('');
  if (lastFlagsHtml !== flagsHtml) $('flags').innerHTML = lastFlagsHtml = flagsHtml; // don't replay the pop-in on every update

  // Advice is fixed, never LLM-written: a small model once told users to "follow the caller's instructions".
  $('ai').hidden = !state.transcript;
  const p = state.pattern;
  $('ai-match').textContent = notice ? 'Looks like a normal one-time-code notice. Never give this code to anyone who asks for it.'
    : !p ? ''
    : p.memoryRisk >= 35 ? `Sounds like ${/^[aeiou]/i.test(p.type) ? 'an' : 'a'} ${p.type}: it matches known scam scripts.`
    : p.risk >= 35 ? 'Resembles the scam calls and texts the model was trained on.'
    : 'Sounds like an ordinary conversation, not a known scam.';
  $('ai-reason').textContent = state.ai?.reason || '';
  $('ai-advice').textContent = ADVICE[level];
  $('transcript').textContent = state.transcript || '—';
}

$('check').onclick = () => {
  const text = $('message').value.trim();
  if (!text) return;
  reset();
  addTranscript(text);
};

function reset() {
  Object.assign(state, { run: state.run + 1, transcript: '', ai: null, pattern: null, seenFlags: new Set() });
  fileChunks = []; fileTotal = 0;
  $('rec').hidden = true;
  render();
}
$('reset').onclick = reset;

function alertBar(msg) {
  $('alert').textContent = msg;
  $('alert').hidden = false;
}

// ---------- Offline indicator + app-shell caching ----------
const net = () => {
  $('net').textContent = navigator.onLine ? '● Online — AI still runs on this phone' : '✈ Offline — 100% on-device';
  $('net').classList.toggle('off', !navigator.onLine);
};
addEventListener('online', net);
addEventListener('offline', net);
net();

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register('./sw.js');
  // The first visit loads before the SW takes control, so cache what this page already fetched.
  addEventListener('load', async () => {
    const urls = performance.getEntriesByType('resource').map(e => e.name).filter(u => u.startsWith(location.origin));
    const cache = await caches.open('shell');
    await Promise.all([location.pathname, ASR_URL.href, EMBED_URL.href, 'icon.svg', 'manifest.webmanifest', ...new Set(urls)].map(u => cache.add(u).catch(() => {})));
  });
}

// Models already downloaded once? Load them from cache straight away instead of asking again.
try { if (localStorage.getItem('scamear-models')) startModels(); } catch {}

// "Share → ScamEar" from the call log or file manager: sw.js stashes the file and opens ./?shared=1.
if (params.has('shared')) {
  history.replaceState(null, '', location.pathname);
  caches.open('share').then(async cache => {
    const res = await cache.match('shared-audio');
    if (!res) return;
    await cache.delete('shared-audio');
    startModels();
    analyseRecording(await res.blob());
  });
}

render();
