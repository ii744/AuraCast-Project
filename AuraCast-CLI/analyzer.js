#!/usr/bin/env node

/**
 * AuraCast Batch Analyzer
 *
 * Provides multi-feature statistical and spectral analysis for detecting
 * AI-generated audio versus authentic human speech.
 *
 * Supported formats: WAV (native), MP3, OGG, FLAC (requires ffmpeg)
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

/**
 * ANSI Color codes for terminal output
 */
const COLORS = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
  magenta: '\x1b[35m',
  bgRed: '\x1b[41m',
  bgGreen: '\x1b[42m',
  bgYellow: '\x1b[43m',
};

const SUPPORTED_EXTENSIONS = new Set(['.wav', '.mp3', '.ogg', '.flac', '.oga', '.opus']);
const FRAME_SECONDS = 0.04;
const LONG_WINDOW_SECONDS = 0.4;
const AI_THRESHOLD = 0.35;

const args = process.argv.slice(2);
const isVerbose = args.includes('--verbose') || args.includes('-v');
const pathArgs = args.filter(arg => !arg.startsWith('-'));

if (pathArgs.length === 0) {
  console.log(`${COLORS.yellow}Usage: node analyzer.js [--verbose] <path_to_audio_file_or_folder>${COLORS.reset}`);
  console.log(`${COLORS.dim}Example: node analyzer.js ./my_audio_folder${COLORS.reset}`);
  console.log(`${COLORS.dim}Example: node analyzer.js ./audio.wav${COLORS.reset}`);
  process.exit(1);
}

const targetPath = pathArgs[0];

/**
 * Performs an in-place Radix-2 Cooley-Tukey FFT.
 * @param {Float64Array} real - Real components
 * @param {Float64Array} imag - Imaginary components
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

/* Detection Algorithm v2 — identical to the web app (script.js).
 *
 * v1 defects fixed here:
 * 1) Speech gate used p10*3 of block RMS — for continuous speech the
 *    threshold rose ABOVE the signal and the analyzer returned
 *    "insufficient speech" (or silently dropped most of the audio).
 * 2) Spectral flatness skipped FFT bins below 1e-10 — silence-like
 *    frequency bands (resampled / band-limited files, e.g. mp3 with a
 *    16 kHz cutoff in a 44.1 kHz container) were excluded, so a HUMAN
 *    voice read as "extremely low flatness" and got flagged as AI.
 * 3) The score was a weighted sum of capped sub-scores that could
 *    never reach 1.0, then shown to the user as "% AI".
 *
 * v2 features (sample-rate invariant):
 *   fluxMean / fluxCV  — spectral change between adjacent 40 ms frames
 *                        (50–4000 Hz band, normalized)
 *   bandFlatCV         — variation of spectral flatness in speech band
 *   energyCV           — loudness variation of 40 ms speech frames
 *   energyCVLong       — loudness variation of 0.4 s blocks
 * Verdict threshold 0.35; aiProbability = logistic(score) for display.
 */

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
 * Analyzes audio data to detect AI generation markers.
 * @param {Float32Array} data - Raw audio samples
 * @param {number} sampleRate - Audio sample rate
 * @returns {Object} Analysis results and metadata
 */
function runDetection(data, sampleRate) {
  const duration = data.length / sampleRate;

  if (duration < 2.0) {
    return createResult('ERROR', 0, 0, duration, 'Audio segment too short for reliable analysis (minimum 2 seconds)', {});
  }

  const frameSamples = Math.max(64, Math.round(sampleRate * FRAME_SECONDS));
  const frameRms = windowRms(data, frameSamples);
  const speechThreshold = speechGate(frameRms);
  const isSpeech = Array.from(frameRms, v => v > speechThreshold);
  const speechCount = isSpeech.filter(Boolean).length;

  if (speechCount < 5) {
    return createResult('ERROR', 0, 0, duration, 'Insufficient speech content detected', {});
  }

  const speechRms = Array.from(frameRms).filter((_, i) => isSpeech[i]);
  const energyCV = calculateCV(speechRms);

  const longSamples = Math.round(sampleRate * LONG_WINDOW_SECONDS);
  const longRms = windowRms(data, longSamples);
  const longSpeechValues = Array.from(longRms).filter(v => v > speechThreshold);
  const energyCVLong = longSpeechValues.length >= 3 ? calculateCV(longSpeechValues) : null;

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

  const features = { energyCV, energyCVLong, fluxMean, fluxCV, bandFlatCV };
  return createResult(verdict, finalScore, aiProbability, duration, details, features, subScores, speechCount);
}

/**
 * Creates a standardized result object.
 */
function createResult(verdict, score, aiProbability, duration, details, features, scores, speechFrames) {
  return {
    verdict,
    score,
    aiProbability,
    duration,
    details,
    features,
    scores: scores || {},
    totalSpeechFrames: speechFrames || 0,
  };
}

/**
 * Math utility functions
 */
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

/**
 * System utility functions
 */
function checkFfmpegAvailability() {
  try {
    execSync('ffmpeg -version', { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

/**
 * Recursively scans directory for audio files.
 */
function scanDirectory(dir, hasFfmpeg) {
  const files = [];
  (function walk(currentDir) {
    for (const entry of fs.readdirSync(currentDir, { withFileTypes: true })) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        if (!entry.name.startsWith('.') && entry.name !== 'node_modules' && entry.name !== 'results') {
          walk(fullPath);
        }
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (SUPPORTED_EXTENSIONS.has(ext) && (ext === '.wav' || hasFfmpeg)) {
          files.push(fullPath);
        }
      }
    }
  })(dir);
  return files.sort();
}

/**
 * Analyzes a single audio file.
 */
function analyzeFile(filePath, hasFfmpeg) {
  const ext = path.extname(filePath).toLowerCase();
  let samples, sampleRate;

  if (ext === '.wav') {
    const result = parseWav(fs.readFileSync(filePath));
    samples = result.samples;
    sampleRate = result.sampleRate;
  } else {
    if (!hasFfmpeg) throw new Error(`ffmpeg is required to process ${ext} files`);
    const result = convertAndParse(filePath);
    samples = result.samples;
    sampleRate = result.sampleRate;
  }

  return runDetection(samples, sampleRate);
}

/**
 * Parses a standard WAV file.
 */
function parseWav(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  if (String.fromCharCode(buffer[0], buffer[1], buffer[2], buffer[3]) !== 'RIFF') {
    throw new Error('Not a valid RIFF file');
  }

  let offset = 12;
  let audioFormat, numChannels, sampleRate, bitsPerSample;
  let formatFound = false;

  while (offset < buffer.length - 8) {
    const chunkId = String.fromCharCode(buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3]);
    const chunkSize = view.getUint32(offset + 4, true);

    if (chunkId === 'fmt ') {
      audioFormat = view.getUint16(offset + 8, true);
      numChannels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bitsPerSample = view.getUint16(offset + 22, true);
      formatFound = true;
    }

    if (chunkId === 'data' && formatFound) {
      const dataOffset = offset + 8;
      const sampleCount = Math.floor(chunkSize / ((bitsPerSample / 8) * numChannels));
      const samples = new Float32Array(sampleCount);

      for (let i = 0; i < sampleCount; i++) {
        const byteOffset = dataOffset + i * numChannels * (bitsPerSample / 8);
        if (byteOffset + (bitsPerSample / 8) > buffer.length) break;

        if (audioFormat === 1) { // PCM
          if (bitsPerSample === 16) samples[i] = view.getInt16(byteOffset, true) / 32768.0;
          else if (bitsPerSample === 24) {
            let val = (buffer[byteOffset + 2] << 16) | (buffer[byteOffset + 1] << 8) | buffer[byteOffset];
            if (val >= 0x800000) val -= 0x1000000;
            samples[i] = val / 8388608.0;
          } else if (bitsPerSample === 32) samples[i] = view.getInt32(byteOffset, true) / 2147483648.0;
        } else if (audioFormat === 3) { // IEEE Float
          samples[i] = view.getFloat32(byteOffset, true);
        }
      }
      return { samples, sampleRate };
    }
    offset += 8 + chunkSize;
    if (chunkSize % 2 !== 0) offset++;
  }
  throw new Error('No data chunk found in WAV file');
}

/**
 * Converts non-WAV formats using ffmpeg.
 */
function convertAndParse(filePath) {
  const tempPath = path.join(os.tmpdir(), `auracast-${Date.now()}.wav`);
  try {
    execSync(`ffmpeg -y -i "${filePath}" -ac 1 -ar 44100 -sample_fmt s16 "${tempPath}" 2>/dev/null`, {
      stdio: 'pipe',
      timeout: 60000,
    });
    return parseWav(fs.readFileSync(tempPath));
  } finally {
    try { fs.unlinkSync(tempPath); } catch (e) { /* ignore cleanup errors */ }
  }
}

/**
 * Main execution entry point.
 */
function main() {
  console.log(`\n${COLORS.bold}${COLORS.magenta}  AuraCast Batch Analyzer${COLORS.reset}\n`);

  const hasFfmpeg = checkFfmpegAvailability();
  if (!hasFfmpeg) {
    console.log(`${COLORS.yellow}  [Note] ffmpeg not found. Only WAV files are supported.${COLORS.reset}\n`);
  }

  const absolutePath = path.resolve(targetPath);
  if (!fs.existsSync(absolutePath)) {
    console.error(`${COLORS.red}  Error: Path not found at ${absolutePath}${COLORS.reset}`);
    process.exit(1);
  }

  const stat = fs.statSync(absolutePath);
  const files = stat.isFile() ? [absolutePath] : scanDirectory(absolutePath, hasFfmpeg);

  if (!files.length) {
    console.log(`${COLORS.yellow}  No supported audio files found.${COLORS.reset}`);
    process.exit(0);
  }

  console.log(`${COLORS.cyan}  Found ${files.length} audio file(s)${COLORS.reset}\n`);

  const results = [];
  for (let i = 0; i < files.length; i++) {
    const filePath = files[i];
    const relativeName = path.relative(absolutePath, filePath) || path.basename(filePath);

    process.stdout.write(`  ${COLORS.dim}[${i + 1}/${files.length}]${COLORS.reset} ${COLORS.white}${relativeName}${COLORS.reset} `);

    try {
      const result = analyzeFile(filePath, hasFfmpeg);
      result.file = relativeName;
      results.push(result);

      let label;
      if (result.verdict === 'AI') {
        label = `${COLORS.bgRed}${COLORS.white}${COLORS.bold} AI ${COLORS.reset}`;
      } else if (result.verdict === 'HUMAN') {
        label = `${COLORS.bgGreen}${COLORS.white}${COLORS.bold} HUMAN ${COLORS.reset}`;
      } else {
        label = `${COLORS.bgYellow}${COLORS.white}${COLORS.bold} ERROR ${COLORS.reset}`;
      }

      const pct = result.verdict === 'ERROR' ? '—' : `${(result.aiProbability * 100).toFixed(1)}%`;
      console.log(`${label} ${COLORS.dim}(${pct})${COLORS.reset}`);

      if (isVerbose && result.features) {
        const { features: f, scores: s } = result;
        const fLong = f.energyCVLong === null ? 'n/a' : f.energyCVLong.toFixed(3);
        console.log(`    ${COLORS.dim}Features: fluxMean=${f.fluxMean?.toFixed(3)} fluxCV=${f.fluxCV?.toFixed(3)} flatCV=${f.bandFlatCV?.toFixed(3)} eCV=${f.energyCV?.toFixed(3)} eCVlong=${fLong}`);
        console.log(`    Scores:   flux=${s.flux?.toFixed(2)} flatStab=${s.flatStability?.toFixed(2)} eJitter=${s.energyJitter?.toFixed(2)} phraseFlat=${s.phraseFlat?.toFixed(2)} fluxStab=${s.fluxStability?.toFixed(2)} → total=${result.score.toFixed(3)}${COLORS.reset}`);
      }
    } catch (err) {
      console.log(`${COLORS.red}ERROR: ${err.message}${COLORS.reset}`);
      results.push({ file: relativeName, verdict: 'ERROR', score: 0, error: err.message });
    }
  }

  console.log('');
  displayTable(results);
  displaySummary(results);
  exportReport(results, absolutePath);
}

/**
 * Formats and displays the results table.
 */
function displayTable(results) {
  const fileNameWidth = Math.max(20, ...results.map(r => r.file.length)) + 2;
  const separator = '─'.repeat(fileNameWidth + 55);

  console.log(`${COLORS.dim}${separator}${COLORS.reset}`);
  console.log(`${COLORS.bold}  ${'File'.padEnd(fileNameWidth)}${'Verdict'.padEnd(10)}${'Score'.padEnd(10)}${'Duration'.padEnd(10)}Details${COLORS.reset}`);
  console.log(`${COLORS.dim}${separator}${COLORS.reset}`);

  for (const r of results) {
    const scoreText = r.verdict === 'ERROR' ? '—' : `${((r.aiProbability ?? r.score) * 100).toFixed(1)}%`;
    const durationText = r.duration ? formatDuration(r.duration) : '—';
    const detailText = (r.details || r.error || '').substring(0, 40);

    let verdictLabel;
    if (r.verdict === 'AI') verdictLabel = `${COLORS.red}${COLORS.bold}AI${COLORS.reset}`.padEnd(19);
    else if (r.verdict === 'HUMAN') verdictLabel = `${COLORS.green}${COLORS.bold}HUMAN${COLORS.reset}`.padEnd(19);
    else verdictLabel = `${COLORS.yellow}ERROR${COLORS.reset}`.padEnd(19);

    console.log(`  ${COLORS.white}${r.file.padEnd(fileNameWidth)}${COLORS.reset}${verdictLabel}${scoreText.padEnd(10)}${durationText.padEnd(10)}${COLORS.dim}${detailText}${COLORS.reset}`);
  }
  console.log(`${COLORS.dim}${separator}${COLORS.reset}`);
}

/**
 * Displays summary statistics by folder.
 */
function displaySummary(results) {
  console.log(`\n${COLORS.bold}${COLORS.cyan}  Summary Statistics${COLORS.reset}\n`);

  const groups = {};
  for (const r of results) {
    if (r.verdict === 'ERROR') continue;
    const folder = path.dirname(r.file);
    if (!groups[folder]) groups[folder] = { total: 0, aiCount: 0, humanCount: 0 };
    groups[folder].total++;
    if (r.verdict === 'AI') groups[folder].aiCount++;
    else groups[folder].humanCount++;
  }

  for (const [folder, g] of Object.entries(groups)) {
    let folderDisplay = folder === '.' ? 'Selected File(s)' : `${folder}/`;
    console.log(`  📁 ${COLORS.white}${folderDisplay}${COLORS.reset} | Total: ${g.total} | Human: ${COLORS.green}${g.humanCount}${COLORS.reset} | AI: ${COLORS.red}${g.aiCount}${COLORS.reset}\n`);
  }

  const validResults = results.filter(r => r.verdict !== 'ERROR');
  console.log(`  ${COLORS.dim}──────────────────────────────────${COLORS.reset}`);
  console.log(`  Total Scanned:  ${validResults.length}`);
  console.log(`  AI Detected:    ${COLORS.red}${validResults.filter(r => r.verdict === 'AI').length}${COLORS.reset}`);
  console.log(`  Human Detected: ${COLORS.green}${validResults.filter(r => r.verdict === 'HUMAN').length}${COLORS.reset}`);

  const errorCount = results.filter(r => r.verdict === 'ERROR').length;
  if (errorCount) console.log(`  Errors encountered: ${COLORS.yellow}${errorCount}${COLORS.reset}`);
  console.log('');
}

/**
 * Saves analysis results to a JSON report.
 */
function exportReport(results, basePath) {
  const outputDir = path.join(__dirname, 'results');
  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const reportPath = path.join(outputDir, `report-${timestamp}.json`);

  const reportData = {
    timestamp: new Date().toISOString(),
    basePath,
    totalFiles: results.length,
    results: results.map(r => ({
      file: r.file,
      verdict: r.verdict,
      score: r.score,
      aiProbability: r.aiProbability,
      duration: r.duration,
      details: r.details,
      features: r.features,
      scores: r.scores,
    })),
  };

  fs.writeFileSync(reportPath, JSON.stringify(reportData, null, 2));
  console.log(`${COLORS.dim}  Report generated: ${reportPath}${COLORS.reset}\n`);
}

/**
 * Helper to format duration in seconds to M:S.
 */
function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

main();
