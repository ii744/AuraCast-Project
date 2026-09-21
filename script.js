/**
 * AuraCast Web Application
 * AI voice detection (FFT-based) and ultrasonic audio encryption.
 */

const ALL_SCREENS = [
  'main-screen', 'encrypt-screen', 'detect-screen',
  'scanning-screen', 'result-human', 'result-ai', 'result-encrypted', 'result-error'
];

let currentEncryptedBlobUrl = null;

function showToast(message) {
  const toast = document.getElementById('toast-container');
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add('show');
  setTimeout(() => {
    toast.classList.remove('show');
  }, 4000);
}

function showScreen(screenId) {
  ALL_SCREENS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = (id === screenId) ? 'block' : 'none';
  });

  if (screenId === 'main-screen' && currentEncryptedBlobUrl) {
    URL.revokeObjectURL(currentEncryptedBlobUrl);
    currentEncryptedBlobUrl = null;
    const player = document.getElementById('encrypted-audio-player');
    if (player) player.src = '';
  }
}

/* ═══════════════════════════════════════════════════════════
   FFT — Radix-2 Cooley-Tukey (in-place)
   ═══════════════════════════════════════════════════════════ */

/**
 * Performs an in-place Radix-2 Cooley-Tukey FFT.
 */
function performFFT(real, imag) {
  const n = real.length;

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = -2 * Math.PI / len;
    const wReal = Math.cos(angle);
    const wImag = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curReal = 1;
      let curImag = 0;
      for (let j = 0; j < halfLen; j++) {
        const uReal = real[i + j];
        const uImag = imag[i + j];
        const vReal = real[i + j + halfLen] * curReal - imag[i + j + halfLen] * curImag;
        const vImag = real[i + j + halfLen] * curImag + imag[i + j + halfLen] * curReal;

        real[i + j] = uReal + vReal;
        imag[i + j] = uImag + vImag;
        real[i + j + halfLen] = uReal - vReal;
        imag[i + j + halfLen] = uImag - vImag;

        const nextReal = curReal * wReal - curImag * wImag;
        curImag = curReal * wImag + curImag * wReal;
        curReal = nextReal;
      }
    }
  }
}

/* ═══════════════════════════════════════════════════════════
   Math Utilities
   ═══════════════════════════════════════════════════════════ */

function calculateMean(arr) {
  if (!arr.length) return 0;
  return arr.reduce((sum, val) => sum + val, 0) / arr.length;
}

function calculateStdDev(arr) {
  if (arr.length < 2) return 0;
  const mean = calculateMean(arr);
  const variance = arr.reduce((sum, val) => sum + (val - mean) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(variance);
}

function calculateCV(arr) {
  const mean = calculateMean(arr);
  return mean > 0 ? calculateStdDev(arr) / mean : 0;
}

/* ═══════════════════════════════════════════════════════════
   Detection Algorithm v2 (calibrated against real TTS and
   microphone recordings)

   v1 had three fatal defects:
   1) Speech gate used p10*3 of block RMS — for continuous speech
      (little silence) the threshold rose ABOVE the signal and the
      analyzer returned "insufficient speech" or silently dropped
      most of the audio.
   2) Spectral flatness skipped FFT bins below 1e-10 — silence-like
      frequency bands (typical for resampled / band-limited files,
      e.g. 8 kHz phone audio inside a 44.1 kHz container) were
      excluded, so a HUMAN voice read as "extremely low flatness"
      and got flagged as AI.
   3) The score was a weighted sum of capped sub-scores that could
      never reach 1.0, then shown to the user as "% AI".

   v2 features (all sample-rate invariant):
     fluxMean / fluxCV  — spectral change between adjacent 40 ms
                          frames (50–4000 Hz band, normalized)
     bandFlatCV         — variation of spectral flatness inside the
                          speech band over time
     energyCV           — loudness variation of 40 ms speech frames
     energyCVLong       — loudness variation of 0.4 s blocks

   Sub-scores are linear ramps calibrated on macOS `say` TTS,
   OpenAI TTS voices and real microphone speech. Verdict threshold
   0.35; raw score is mapped to a percentage through a logistic
   curve so honest AI audio displays 85–99% instead of ~64%.
   ═══════════════════════════════════════════════════════════ */

const FRAME_SECONDS = 0.04;
const LONG_WINDOW_SECONDS = 0.4;
const AI_THRESHOLD = 0.35;

function clamp01(x) {
  return Math.min(Math.max(x, 0), 1);
}

function median(sortedArr) {
  const n = sortedArr.length;
  if (!n) return 0;
  const mid = Math.floor(n / 2);
  return n % 2 ? sortedArr[mid] : (sortedArr[mid - 1] + sortedArr[mid]) / 2;
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/* RMS for non-overlapping windows of `win` samples */
function windowRms(data, win) {
  const count = Math.floor(data.length / win);
  const rms = new Float64Array(count);
  for (let w = 0; w < count; w++) {
    let sum = 0;
    const base = w * win;
    for (let i = 0; i < win; i++) sum += data[base + i] * data[base + i];
    rms[w] = Math.sqrt(sum / win);
  }
  return rms;
}

/* Fixed speech gate: works for continuous speech (v1 failed there) */
function speechGate(rmsValues) {
  const sorted = Array.from(rmsValues).sort((a, b) => a - b);
  const p10 = sorted[Math.floor(sorted.length * 0.1)];
  const med = median(sorted);
  return Math.max(0.003, Math.min(p10 * 2, med * 0.45));
}

/**
 * Analyzes raw PCM data to detect AI markers using spectral and statistical features.
 */
function runDetection(data, sampleRate) {
  const detectorVersion = 'v3';

  const duration = data.length / sampleRate;
  if (duration < 2.0) {
    return { verdict: 'ERROR', score: 0, aiProbability: 0, details: 'Audio too short for reliable analysis (minimum 2 seconds)' };
  }

  const frameSamples = Math.max(64, Math.round(sampleRate * FRAME_SECONDS));
  const frameRms = windowRms(data, frameSamples);
  const speechThreshold = speechGate(frameRms);
  const isSpeech = Array.from(frameRms, v => v > speechThreshold);
  const speechCount = isSpeech.filter(Boolean).length;

  if (speechCount < 5) {
    return { verdict: 'ERROR', score: 0, aiProbability: 0, details: 'Insufficient speech content detected' };
  }

  /* --- Feature 1: energy variation, 40 ms frames --- */
  const speechRms = Array.from(frameRms).filter((_, i) => isSpeech[i]);
  const energyCV = calculateCV(speechRms);

  /* --- Feature 2: energy variation, 0.4 s blocks (phrase level) --- */
  const longSamples = Math.round(sampleRate * LONG_WINDOW_SECONDS);
  const longRms = windowRms(data, longSamples);
  const longSpeechValues = Array.from(longRms).filter(v => v > speechThreshold);
  const energyCVLong = longSpeechValues.length >= 3 ? calculateCV(longSpeechValues) : null;

  /* --- Features 3-4: spectral flux and flatness inside speech band --- */
  const fftSize = nextPow2(frameSamples);
  const binHz = sampleRate / fftSize;
  const loBin = Math.max(1, Math.floor(50 / binHz));
  const hiBin = Math.min(fftSize / 2 - 1, Math.floor(4000 / binHz));
  const bandBins = hiBin - loBin + 1;

  const real = new Float64Array(fftSize);
  const imag = new Float64Array(fftSize);
  const hann = new Float64Array(frameSamples);
  for (let i = 0; i < frameSamples; i++) {
    hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (frameSamples - 1));
  }

  const bandMagnitudes = [];
  const flatnesses = [];

  for (let f = 0; f < frameRms.length; f++) {
    if (!isSpeech[f]) continue;

    real.fill(0);
    imag.fill(0);
    const base = f * frameSamples;
    for (let i = 0; i < frameSamples; i++) {
      real[i] = data[base + i] * hann[i];
    }
    performFFT(real, imag);

    const mag = new Float64Array(bandBins);
    let sum = 0;
    for (let k = loBin; k <= hiBin; k++) {
      const m = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
      mag[k - loBin] = m;
      sum += m;
    }
    if (sum > 0) {
      const normed = new Float64Array(bandBins);
      for (let i = 0; i < bandBins; i++) normed[i] = mag[i] / sum;
      bandMagnitudes.push(normed);
    }

    let logSum = 0;
    for (let i = 0; i < bandBins; i++) logSum += Math.log(mag[i] + 1e-12);
    flatnesses.push(Math.exp(logSum / bandBins) / (sum / bandBins + 1e-12));
  }

  const fluxes = [];
  for (let i = 1; i < bandMagnitudes.length; i++) {
    let diffSum = 0;
    const cur = bandMagnitudes[i];
    const prev = bandMagnitudes[i - 1];
    for (let j = 0; j < bandBins; j++) {
      const d = cur[j] - prev[j];
      diffSum += d * d;
    }
    fluxes.push(Math.sqrt(diffSum));
  }

  const fluxMean = calculateMean(fluxes);
  const fluxCV = fluxes.length > 1 ? calculateCV(fluxes) : 0.5;
  const bandFlatCV = flatnesses.length > 1 ? calculateCV(flatnesses) : 0.5;

  /* --- Calibrated sub-scores (linear ramps) --- */
  const subScores = {
    flux: clamp01((fluxMean - 0.133) / 0.035),
    flatStability: clamp01((bandFlatCV - 0.55) / 0.20),
    energyJitter: clamp01((0.70 - energyCV) / 0.25),
    phraseFlat: energyCVLong === null ? 0.5 : clamp01((0.28 - energyCVLong) / 0.16),
    fluxStability: clamp01((fluxCV - 0.41) / 0.05),
  };

  /* v3 weights: flux / flatStability get low weight — on phone recordings
     (Voice Memos DSP, AAC) they fire for real human voices. The main
     load is carried by loudness dynamics (energyJitter, phraseFlat) and
     the regularity of spectral transitions (fluxStability). */
  const weights = {
    flux: 0.10,
    flatStability: 0.10,
    energyJitter: 0.30,
    phraseFlat: 0.20,
    fluxStability: 0.30,
  };

  let finalScore = 0;
  for (const [key, weight] of Object.entries(weights)) {
    finalScore += subScores[key] * weight;
  }

  const verdict = finalScore >= AI_THRESHOLD ? 'AI' : 'HUMAN';
  const aiProbability = 1 / (1 + Math.exp(-(finalScore - AI_THRESHOLD) * 12));

  const indicators = [];
  if (subScores.flux > 0.4) indicators.push('unnaturally rapid spectral change');
  if (subScores.flatStability > 0.4) indicators.push('unstable spectral profile');
  if (subScores.energyJitter > 0.4) indicators.push('overly uniform loudness');
  if (subScores.phraseFlat > 0.4) indicators.push('phrase-level energy too even');
  if (subScores.fluxStability > 0.4) indicators.push('repetitive spectral transitions');

  let details = indicators.length > 0 ? indicators.join(', ') : 'Authentic speech patterns';
  if (finalScore > 0.25 && finalScore < 0.50) {
    details += ' (low certainty — short or borderline sample)';
  }

  return {
    verdict,
    score: finalScore,
    aiProbability,
    details,
    detectorVersion,
    features: { energyCV, energyCVLong, fluxMean, fluxCV, bandFlatCV, sampleRate },
    subScores,
  };
}

/* ═══════════════════════════════════════════════════════════
   AI Detect Handler
   ═══════════════════════════════════════════════════════════ */

async function handleDetect(inputElement) {
  const file = inputElement.files[0];
  if (!file) return;

  showScreen('scanning-screen');
  const scanLabel = document.getElementById('scanning-label');
  scanLabel.textContent = `Analyzing: ${file.name}`;

  let audioCtx = null;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    
    if (audioBuffer.duration < 2.0) {
      showToast('Audio file is too short (minimum 2 seconds required).');
      showScreen('detect-screen');
      return;
    }

    const data = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;

    const result = await new Promise(resolve => {
      setTimeout(() => resolve(runDetection(data, sampleRate)), 50);
    });

    const confidencePercent = Math.round(result.aiProbability * 100);

    if (result.verdict === 'ERROR') {
      document.getElementById('error-details').textContent = result.details;
      showScreen('result-error');
    } else if (result.verdict === 'AI') {
      document.getElementById('ai-confidence-value').textContent = `${confidencePercent}%`;
      document.getElementById('ai-confidence-bar').style.width = `${confidencePercent}%`;
      document.getElementById('ai-details').textContent = result.details;
      showScreen('result-ai');
    } else {
      const humanConfidence = 100 - confidencePercent;
      document.getElementById('human-confidence-value').textContent = `${humanConfidence}%`;
      document.getElementById('human-confidence-bar').style.width = `${humanConfidence}%`;
      document.getElementById('human-details').textContent = result.details;
      showScreen('result-human');
    }

  } catch (e) {
    console.error('Detection error:', e);
    alert('Error reading audio file.');
    showScreen('detect-screen');
  } finally {
    if (audioCtx) audioCtx.close();
    inputElement.value = '';
  }
}

/* ═══════════════════════════════════════════════════════════
   Audio Encryption (Ultrasonic Watermark)
   ═══════════════════════════════════════════════════════════ */

/**
 * Encodes an AudioBuffer into a valid 16-bit PCM WAV blob.
 */
function createWavFile(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleCount = buffer.length;
  const dataSize = sampleCount * numChannels * 2;
  const fileSize = 44 + dataSize;
  const arrayBuffer = new ArrayBuffer(fileSize);
  const view = new DataView(arrayBuffer);

  let offset = 0;

  function writeUint16(val) { view.setUint16(offset, val, true); offset += 2; }
  function writeUint32(val) { view.setUint32(offset, val, true); offset += 4; }

  writeUint32(0x46464952); // "RIFF"
  writeUint32(fileSize - 8);
  writeUint32(0x45564157); // "WAVE"
  writeUint32(0x20746d66); // "fmt "
  writeUint32(16);
  writeUint16(1); // PCM format
  writeUint16(numChannels);
  writeUint32(buffer.sampleRate);
  writeUint32(buffer.sampleRate * numChannels * 2);
  writeUint16(numChannels * 2);
  writeUint16(16); // 16-bit
  writeUint32(0x61746164); // "data"
  writeUint32(dataSize);

  const channels = [];
  for (let ch = 0; ch < numChannels; ch++) {
    channels.push(buffer.getChannelData(ch));
  }

  for (let i = 0; i < sampleCount; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = Math.max(-1, Math.min(1, channels[ch][i]));
      const int16 = sample < 0 ? sample * 32768 : sample * 32767;
      view.setInt16(offset, int16 | 0, true);
      offset += 2;
    }
  }

  return new Blob([arrayBuffer], { type: 'audio/wav' });
}

async function handleEncrypt(inputElement) {
  const file = inputElement.files[0];
  if (!file) return;

  const uploadText = inputElement.previousElementSibling;
  const originalText = uploadText.innerText;
  uploadText.innerText = 'Processing...';
  inputElement.parentElement.style.pointerEvents = 'none';
  inputElement.parentElement.style.opacity = '0.5';

  let audioCtx = null;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    const originalBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const offlineCtx = new OfflineAudioContext(
      originalBuffer.numberOfChannels,
      originalBuffer.length,
      originalBuffer.sampleRate
    );

    const source = offlineCtx.createBufferSource();
    source.buffer = originalBuffer;

    const oscillator1 = offlineCtx.createOscillator();
    oscillator1.type = 'sine';
    oscillator1.frequency.value = 18500;

    const oscillator2 = offlineCtx.createOscillator();
    oscillator2.type = 'square';
    oscillator2.frequency.value = 19500;

    const gainNode = offlineCtx.createGain();
    gainNode.gain.value = 0.15;

    source.connect(offlineCtx.destination);
    oscillator1.connect(gainNode);
    oscillator2.connect(gainNode);
    gainNode.connect(offlineCtx.destination);

    source.start(0);
    oscillator1.start(0);
    oscillator2.start(0);

    const renderedBuffer = await offlineCtx.startRendering();
    const encryptedBlob = createWavFile(renderedBuffer);

    if (currentEncryptedBlobUrl) {
      URL.revokeObjectURL(currentEncryptedBlobUrl);
    }
    currentEncryptedBlobUrl = URL.createObjectURL(encryptedBlob);

    const player = document.getElementById('encrypted-audio-player');
    player.src = currentEncryptedBlobUrl;
    player.load();
    showScreen('result-encrypted');

  } catch (e) {
    console.error('Encryption error:', e);
    alert('Error encrypting audio.');
  } finally {
    if (audioCtx) audioCtx.close();
    uploadText.innerText = originalText;
    inputElement.parentElement.style.pointerEvents = 'auto';
    inputElement.parentElement.style.opacity = '1';
    inputElement.value = '';
  }
}

/* ═══════════════════════════════════════════════════════════
   Simulate AI Attack
   ═══════════════════════════════════════════════════════════ */

async function simulateAIListening() {
  if (!currentEncryptedBlobUrl) return;

  document.getElementById('encrypted-audio-player').pause();

  let audioCtx = null;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const response = await fetch(currentEncryptedBlobUrl);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;

    const distortion = audioCtx.createWaveShaper();
    function makeDistortionCurve(amount) {
      const k = typeof amount === 'number' ? amount : 50;
      const samples = 44100;
      const curve = new Float32Array(samples);
      const deg = Math.PI / 180;
      for (let i = 0; i < samples; i++) {
        const x = i * 2 / samples - 1;
        curve[i] = (3 + k) * x * 20 * deg / (Math.PI + k * Math.abs(x));
      }
      return curve;
    }
    distortion.curve = makeDistortionCurve(400);

    const biquadFilter = audioCtx.createBiquadFilter();
    biquadFilter.type = 'highpass';
    biquadFilter.frequency.value = 2000;
    biquadFilter.Q.value = 15;

    source.connect(biquadFilter);
    biquadFilter.connect(distortion);
    distortion.connect(audioCtx.destination);

    showToast('SIMULATION START: Now you will hear how your voice sounds to AI.');
    source.start(0);

    source.onended = () => {
      audioCtx.close();
    };
  } catch (e) {
    console.error('Simulation error:', e);
    if (audioCtx) audioCtx.close();
  }
}
