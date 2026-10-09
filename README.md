# ScamEar 👂🚩

**A scam-call detector that runs 100% on your phone.** 📖 **Full guide (how it works, numbers, judge Q&A): [docs/GUIDE.md](docs/GUIDE.md)**

Put a suspicious caller on speaker near ScamEar and it transcribes the call live, spots scam tactics as they happen, and tells you to hang up. It works in airplane mode.

## Why it has to be local

- **Privacy:** a scam detector has to hear your private calls and read your private messages. Streaming those to a cloud AI is the exact privacy risk it's meant to protect you from.
- **No signal, no cost:** scams target people everywhere, including places with bad data and people who can't pay for cloud AI. Once it's loaded, ScamEar needs zero network and costs nothing per call.
- **Real-time:** red flags appear within seconds of being spoken, with no round-trip to a server.

## How it works

```
mic ─▶ Whisper-small (Web Worker, multi-threaded WASM)
        hears English / Tagalog / Taglish and translates to English on-device
                              │
        ┌─────────────────────┼──────────────────────────┐
        ▼                     ▼                          ▼
  Red-flag rules       multilingual-e5 embedding   Qwen2.5-1.5B (WebGPU phones only)
  EN + TL keywords     ├ scam memory: nearest of   explains the tactic
  OTP, GCash, "wag     │ 54 EN/TL scripts (type)
  sabihin"…            └ trained classifier (9.8k)
        └──────────────▶ risk = max(rules, memory, classifier, LLM) ◀──────┘
                              ▼
           gauge · 🚩 flags · vibration · fixed safety advice
```

- Safety advice is **fixed text, never LLM-generated.** A small model once told the user to "follow the caller's instructions".
- Legit code notices ("Your OTP is 482913. Do not share…", with no link or phone number) are treated as normal. Being *asked* for a code is what gets flagged.
- Works on **any** phone: without WebGPU (e.g. Exynos/Xclipse GPUs), the rules, scam memory and classifier still run on the CPU.
- Multi-threaded WASM needs cross-origin isolation, so the host must send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless` (set in `vite.config.js` for dev and preview).
- Models are cached by the browser after the first load, and the app shell is cached by a service worker, so the app reloads and runs fully offline.

## Checking real phone calls

Android doesn't let third-party apps capture call audio while a call is happening; only the phone's own dialer can. ScamEar works with that in two ways:

1. **Live, on speakerphone:** tap *Start listening* and put the call on speaker (or hold ScamEar near a landline).
2. **After the call, from the recording:** turn on *Phone → Settings → Record calls → Auto record calls* (Samsung). After a suspicious call, tap *Choose a call recording* or **Share → ScamEar** from the call log or *My Files*. ScamEar decodes the `.m4a`, transcribes and translates it in 30-second parts, and scores the whole conversation, offline. A 35-second call takes about 8 seconds on a laptop.

Share → ScamEar needs the app installed from an HTTPS address ("Add to Home screen"); the file picker works everywhere. `demo/` has two sample recordings (a Taglish scam call and a normal family call) for trying it out.

## Disclosure (models, libraries, tools)

| Component | What |
|---|---|
| Speech recognition + translation | [Xenova/whisper-small](https://huggingface.co/Xenova/whisper-small) (OpenAI Whisper, multilingual, ONNX q8) via [Transformers.js](https://github.com/huggingface/transformers.js) on ONNX Runtime Web (WASM) |
| Scam memory (embeddings) | [Xenova/multilingual-e5-small](https://huggingface.co/Xenova/multilingual-e5-small) (ONNX q8), nearest-neighbour vs. `src/patterns.js` |
| Trained scam classifier | Logistic regression on e5 embeddings (`src/classifier.json`, 384 weights), trained by us with `training/train.mjs` |
| Training data | [BothBosu/scam-dialogue](https://huggingface.co/datasets/BothBosu/scam-dialogue) + [BothBosu/multi-agent-scam-conversation](https://huggingface.co/datasets/BothBosu/multi-agent-scam-conversation) (Apache-2.0, synthetic call dialogues); [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (CC BY 4.0); [Philippine Spam/Scam SMS](https://www.kaggle.com/datasets/bwandowando/philippine-spam-sms-messages) by bwandowando (CC BY-NC-SA 4.0, real texts received in the Philippines, 2024–2026); plus our own 54 EN/Tagalog scripts and 80 normal Filipino messages (`training/ph-normal.mjs`) |
| Reasoning LLM (optional) | Qwen2.5-1.5B-Instruct (q4f16/q4f32 MLC build) via [WebLLM](https://github.com/mlc-ai/web-llm), only on phones where Chrome exposes WebGPU |
| Build tool | Vite |
| Cloud APIs | **None.** Hugging Face / MLC CDNs are used only to download model weights on first load. |
| AI-assisted development | Claude Code |

URL options: `?llm=0.5B` for a smaller LLM, `?asr=<model>` to swap the Whisper model.

## Evaluation

Held-out data the model never saw, scored exactly like the app (`node training/stats.mjs`). "Scam" means a score of 35 or more.

| Test set | Scam memory only | Trained classifier only | **Shipped app** |
|---|---|---|---|
| All held-out calls + SMS (n = 3,107) | 81.0% acc, 62% recall | 93.5% acc, 91% recall | **92.3% acc, 94.3% recall** |
| Real Philippine spam/scam SMS (n = 182) | | | **93.4% caught** (86.8% before adding this dataset) |
| Normal Filipino messages (n = 16) | | | 2 false alarms |
| Hand-written EN + Tagalog/Taglish set (n = 25) | 24/25 | 18/25 | **24/25** |

Shipped app on the 3,107 held-out examples: precision 88.2%, recall 94.3%, F1 0.911, false-alarm rate 9.1%. The trained classifier's ROC-AUC is 0.981. The full report includes the confusion matrix and per-scam-type results.

The English public datasets are US-style (SSN, refund, tech support, delivery), and the Philippine set has spam only. So training adds our own Filipino scripts and normal Filipino messages, weighted up so the model doesn't learn "Tagalog + money talk = scam".

## Training

```bash
cd training && mkdir -p csv sms ph-sms
for f in scam-dialogue_train scam-dialogue_test; do curl -sL -o csv/$f.csv https://huggingface.co/datasets/BothBosu/scam-dialogue/resolve/main/$f.csv; done
for f in agent_conversation_train agent_conversation_test; do curl -sL -o csv/$f.csv https://huggingface.co/datasets/BothBosu/multi-agent-scam-conversation/resolve/main/$f.csv; done
curl -sL -o sms/sms.zip "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip" && (cd sms && unzip -o sms.zip)
curl -sL -o ph-sms/ph.zip https://www.kaggle.com/api/v1/datasets/download/bwandowando/philippine-spam-sms-messages && (cd ph-sms && unzip -o ph.zip)
node download.mjs && node train.mjs && node stats.mjs   # ~3 min on a laptop CPU → writes src/classifier.json
```

## Run it

```bash
npm install
npm test          # rule-engine self-check
npm run build && npm run preview
```

Mic, camera and WebGPU require **HTTPS** on a phone, so deploy `dist/` to any static host (Vercel, Netlify, GitHub Pages).

## Demo script (≈2 min)

1. **Before the demo, on WiFi:** open the site in Android Chrome, tap *Download & start*, and wait for 👂 *Ears* and 📚 *Scam memory* to show *Ready* (🧠 only loads on phones with WebGPU). Then add it to your home screen.
2. **On stage:** turn on **airplane mode** and show it. Close and reopen the app: the badge shows *✈ Offline — 100% on-device*.
3. Hand a judge the scam script below and ask them to read it out like a phone call. Tap **Start listening**.
4. The flags pop in one by one, the phone vibrates, the screen pulses red, and the verdict reads **LIKELY SCAM — HANG UP**.
5. **Clear**, then paste a fake "you won a prize" SMS into *Got a suspicious text?* It's flagged instantly, still offline.
6. Close with: *"Nothing you just heard left this phone. Your calls stay private, it costs nothing per call, and it works with no signal."*

### Scam script (print this for the judge)

> "Hello, this is Mark from the fraud department of your bank. We detected suspicious activity and your account will be **frozen today**. To stop this, I need you to read me the **six-digit code we just sent** to your phone. Please do it **right now**, and **don't tell anyone**, this is confidential. If you can't, we can also secure your money if you **transfer it to a safe account**."

## Known limits

- Android doesn't let any third-party app (web or native) capture call audio while the call is happening. ScamEar listens on **speakerphone**, or checks the phone's own **call recording** right after the call. Live in-call warnings need dialer-level access, which only phone makers have (Google's own scam detection lives in its Phone app).
- Speech is translated in ~5 s chunks, so alerts arrive a few seconds after the words are spoken. Translation sometimes drops words like "OTP". The scam memory covers for this by matching meaning, not keywords.
- Pushy telemarketing calls are flagged as suspicious about 39% of the time, and an English "Windows technical department" call scores 30, just under the warning line.
- Training includes CC BY-NC-SA data, so the shipped classifier is for non-commercial use. A commercial version would need to retrain without that dataset or get the author's permission.
