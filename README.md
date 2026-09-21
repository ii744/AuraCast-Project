# AuraCast 🎙️
> **Your voice is your biometrics. Protect it.**

<p align="center">
  <a href="https://infomatrixisland.vercel.app/">
    <img alt="Try Online" src="https://img.shields.io/badge/Try_Online-Live_Demo-0078D4?style=for-the-badge&logo=vercel&logoColor=white">
  </a>
  <a href="https://www.figma.com/design/bkIxLkjMcPS53dXsoPlvqd/AuraCast?node-id=0-1&p=f&t=xSfVlGnW3k9hS2EB-0">
    <img alt="Figma Design" src="https://img.shields.io/badge/Figma-Design-F24E1E?style=for-the-badge&logo=figma&logoColor=white">
  </a>
</p>

AuraCast is a project built around one simple question: how do you know the voice you're hearing is actually human? It comes in two parts — a browser-based web app for everyday users, and a command-line batch analyzer for more technical use cases. Both tackle the same problem from different angles: protecting real voices and exposing fake ones.

All processing happens locally. No audio ever leaves your device.

## ✨ Core Features

### 1. Audio Encryption — Data Poisoning
Prevents AI models from parsing or cloning your voice.

- **How it works:** When you upload an audio file, AuraCast injects high-frequency noise (18.5 kHz–19.5 kHz) into the audio spectrum — a technique known as adversarial perturbation.
- **The result:** The changes are essentially inaudible to humans, but they destroy the feature vectors that AI voice-cloning models rely on. The audio becomes unusable for training or synthesis.
- **Simulation Mode:** There's a built-in "Simulate AI Attack" button that lets you actually *hear* what the poisoned audio sounds like to a neural network — distorted, filtered, broken. It's worth noting that a real AI attack happens entirely in the model's internal representation, which you can't hear at all. The simulation exists purely to give people an intuitive sense of what's going on under the hood.

### 2. AI Voice Detection — Deepfake Scanner
Tells you whether a voice recording is human or AI-generated, without sending anything to a server.

- **How it works:** The scanner uses Digital Signal Processing (DSP) heuristics to analyze the audio signal directly, working on 40 ms frames inside the speech band (50–4000 Hz), so results don't depend on the file's sample rate.
- **What it looks for:**
  - **Spectral flux:** How much the spectrum changes from frame to frame. Human speech changes in a characteristic, organic way; synthesized voices shift too abruptly and too repetitively.
  - **Spectral profile stability:** How much the timbre (spectral flatness inside the speech band) drifts over time. Neural vocoders produce an unnaturally "shimmering" spectral profile.
  - **Energy variance:** Human speech is naturally dynamic — there's breathing, micro-pauses, variation in loudness — at both the 40 ms (syllable) and 0.4 s (phrase) scale. AI-generated voices often have suspiciously stable energy levels.
- The detector was re-calibrated (v3, Sept 2026) against macOS TTS voices, OpenAI TTS voices, and real phone (Voice Memos) recordings at 8/16/44.1/48 kHz. v3 reweights the five sub-scores so that phone recordings (Voice Memos DSP + AAC compression) no longer trip the spectral-flux and flatness-stability cues for real human voices — energy jitter and flux stability now carry most of the weight, while flux and flatness stability are de-weighted. Real voice recordings classify as human and AI-generated recordings classify as AI. One honest limitation: highly expressive neural TTS voices (e.g. OpenAI's nova/onyx) can drift toward the borderline — DSP heuristics can't rival a trained neural detector, so treat borderline results with suspicion.

---

## 🖥️ AuraCast CLI — Batch Analyzer

For situations where you need to process a large number of files at once, there's a separate Node.js command-line tool in the `AuraCast-CLI/` folder.

It uses the same **FFT (Fast Fourier Transform)**-based detection engine as the web app — spectral flux, spectral flatness variation in the speech band, and energy coefficient of variation — calibrated on TTS and real microphone samples (calibration set in v3: macOS/OpenAI TTS plus real phone recordings).

**Supported formats:** WAV (native), MP3, OGG, FLAC, Opus (requires ffmpeg for non-WAV)

### How to run the CLI

```bash
# Navigate to the CLI folder
cd AuraCast-CLI

# Analyze a single file
node analyzer.js path/to/audio.wav

# Analyze an entire folder (recursively)
node analyzer.js path/to/folder/

# Verbose mode — shows feature scores for each file
node analyzer.js --verbose path/to/folder/
```

Results are automatically saved as a JSON report in `AuraCast-CLI/results/`.

**Note:** ffmpeg is required to process MP3, OGG, FLAC, and Opus files. WAV files work out of the box with no dependencies.

---

## 🌐 Running the Web App

The web app is a plain HTML/CSS/JS project — no build step needed.

```bash
# From the repo root, just open index.html in your browser
open index.html

# Or if you prefer a local server (avoids any file:// quirks)
npx serve .
# then go to http://localhost:3000
```

That's it. No npm install, no bundler, no backend.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Web App | Vanilla JS, Web Audio API |
| CLI Analyzer | Node.js (no external dependencies) |
| FFT Implementation | Custom Cooley-Tukey Radix-2 (written from scratch) |
| UI Design | Figma → hand-coded CSS |
| Fonts | Space Grotesk (Google Fonts) |

---

## 🤝 AI Tools & Acknowledgements

This project was built with some help from AI tools, and we think it's important to be upfront about that.

- **AuraCast CLI** (`AuraCast-CLI/analyzer.js`) — The batch analyzer was developed with the assistance of **Claude Code**. The FFT algorithm, spectral feature extraction, and scoring logic were refined iteratively through that process.
- **Web App Design** — The UI/UX was designed entirely by hand in **Figma**. Every layout decision, color choice, and component was made manually.
- **Web App Implementation** — Turning the Figma design into working code, as well as finding and fixing bugs in the JavaScript logic, was done with help from **Gemini**.

The core ideas, problem framing, algorithm selection, and testing were all done by the team.
