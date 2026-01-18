# 007 Pose Web App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a client-side web version of 007 Pose game deployable to GitHub Pages.

**Architecture:** Single-page app using MediaPipe JS for pose detection, TensorFlow.js for classifier and RL opponent, Canvas for rendering, IndexedDB for persistence.

**Tech Stack:** Vanilla JS, MediaPipe Pose (JS), TensorFlow.js, IndexedDB, HTML5 Canvas

---

## Task 1: Project Scaffolding

**Files:**
- Create: `web/index.html`
- Create: `web/css/style.css`
- Create: `web/js/main.js`

**Step 1: Create directory structure**

```bash
mkdir -p web/css web/js web/assets
```

**Step 2: Create index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>007 Pose</title>
  <link rel="stylesheet" href="css/style.css">
</head>
<body>
  <div id="app">
    <header>
      <h1>007 POSE</h1>
      <button id="calibrate-btn">Calibrate</button>
    </header>
    <main>
      <div id="video-container">
        <video id="webcam" autoplay playsinline></video>
        <canvas id="overlay"></canvas>
      </div>
      <div id="game-info">
        <div id="player-info">
          <span class="label">YOU</span>
          <div id="player-bullets" class="bullets"></div>
        </div>
        <div id="round-info">Round <span id="round-num">0</span></div>
        <div id="ai-info">
          <span class="label">AI</span>
          <div id="ai-bullets" class="bullets"></div>
        </div>
      </div>
      <div id="message"></div>
      <div id="score">You: <span id="wins">0</span> - <span id="losses">0</span> AI</div>
    </main>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/pose.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
  <script type="module" src="js/main.js"></script>
</body>
</html>
```

**Step 3: Create style.css**

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #1a1a2e;
  color: #eee;
  min-height: 100vh;
}

#app {
  max-width: 900px;
  margin: 0 auto;
  padding: 1rem;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

h1 {
  color: #00ffff;
  font-size: 1.5rem;
}

button {
  background: #333;
  color: #eee;
  border: 1px solid #555;
  padding: 0.5rem 1rem;
  cursor: pointer;
  border-radius: 4px;
}

button:hover {
  background: #444;
}

#video-container {
  position: relative;
  width: 100%;
  aspect-ratio: 16/9;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
}

#webcam, #overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

#webcam {
  transform: scaleX(-1);
  object-fit: cover;
}

#overlay {
  transform: scaleX(-1);
  pointer-events: none;
}

#game-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 1rem;
  padding: 0.5rem;
}

.label {
  font-size: 0.8rem;
  color: #888;
}

#player-info .label { color: #00ffff; }
#ai-info .label { color: #ff6b6b; }

.bullets {
  font-family: monospace;
  margin-top: 0.25rem;
}

#message {
  text-align: center;
  font-size: 2rem;
  padding: 1rem;
  min-height: 4rem;
}

#score {
  text-align: center;
  color: #888;
  margin-top: 0.5rem;
}
```

**Step 4: Create main.js entry point**

```javascript
// 007 Pose Web App - Main Entry Point

console.log('007 Pose loading...');

// App state
const state = {
  webcamReady: false,
  poseReady: false,
  classifierReady: false,
  gameState: null,
};

async function init() {
  console.log('Initializing app...');
  // TODO: Initialize webcam
  // TODO: Initialize MediaPipe Pose
  // TODO: Initialize classifier
  // TODO: Start game loop
}

init().catch(console.error);
```

**Step 5: Verify it loads**

Open `web/index.html` in browser, check console shows "007 Pose loading..."

**Step 6: Commit**

```bash
git add web/
git commit -m "feat(web): add project scaffolding with HTML, CSS, JS entry point"
```

---

## Task 2: Webcam Setup

**Files:**
- Modify: `web/js/main.js`

**Step 1: Add webcam initialization**

Replace the TODO comment with:

```javascript
async function initWebcam() {
  const video = document.getElementById('webcam');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: 'user' }
    });
    video.srcObject = stream;
    await video.play();

    // Size canvas to match video
    const canvas = document.getElementById('overlay');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    console.log('Webcam initialized:', video.videoWidth, 'x', video.videoHeight);
    return true;
  } catch (err) {
    console.error('Webcam error:', err);
    document.getElementById('message').textContent = 'Camera access required';
    return false;
  }
}
```

**Step 2: Update init function**

```javascript
async function init() {
  console.log('Initializing app...');

  state.webcamReady = await initWebcam();
  if (!state.webcamReady) return;

  document.getElementById('message').textContent = 'Camera ready!';
}
```

**Step 3: Verify webcam shows**

Open in browser, accept camera permission, verify video feed appears.

**Step 4: Commit**

```bash
git add web/js/main.js
git commit -m "feat(web): add webcam initialization"
```

---

## Task 3: MediaPipe Pose Integration

**Files:**
- Create: `web/js/pose-detector.js`
- Modify: `web/js/main.js`

**Step 1: Create pose-detector.js**

```javascript
// MediaPipe Pose wrapper

export class PoseDetector {
  constructor() {
    this.pose = null;
    this.lastLandmarks = null;
    this.onResults = this.onResults.bind(this);
  }

  async init() {
    this.pose = new Pose({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`;
      }
    });

    this.pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    this.pose.onResults(this.onResults);

    // Warm up the model
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    await this.pose.send({ image: canvas });

    console.log('MediaPipe Pose initialized');
  }

  onResults(results) {
    if (results.poseLandmarks) {
      this.lastLandmarks = results.poseLandmarks;
    } else {
      this.lastLandmarks = null;
    }
  }

  async detect(videoElement) {
    await this.pose.send({ image: videoElement });
    return this.lastLandmarks;
  }

  getLandmarks() {
    return this.lastLandmarks;
  }
}
```

**Step 2: Update main.js to use pose detector**

Add at top:
```javascript
import { PoseDetector } from './pose-detector.js';
```

Add to state:
```javascript
const state = {
  webcamReady: false,
  poseReady: false,
  classifierReady: false,
  gameState: null,
  poseDetector: null,
};
```

Add initialization:
```javascript
async function initPose() {
  state.poseDetector = new PoseDetector();
  await state.poseDetector.init();
  console.log('Pose detector ready');
  return true;
}
```

Update init:
```javascript
async function init() {
  console.log('Initializing app...');

  state.webcamReady = await initWebcam();
  if (!state.webcamReady) return;

  document.getElementById('message').textContent = 'Loading pose model...';
  state.poseReady = await initPose();

  document.getElementById('message').textContent = 'Ready!';

  // Start detection loop
  detectLoop();
}

async function detectLoop() {
  const video = document.getElementById('webcam');
  const landmarks = await state.poseDetector.detect(video);

  if (landmarks) {
    drawLandmarks(landmarks);
  }

  requestAnimationFrame(detectLoop);
}

function drawLandmarks(landmarks) {
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw pose dots
  ctx.fillStyle = '#00ff00';
  for (const lm of landmarks) {
    if (lm.visibility > 0.5) {
      ctx.beginPath();
      ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
```

**Step 3: Verify pose detection**

Open in browser, verify green dots appear on body landmarks.

**Step 4: Commit**

```bash
git add web/js/
git commit -m "feat(web): add MediaPipe Pose detection with landmark visualization"
```

---

## Task 4: Pose Classifier (TensorFlow.js)

**Files:**
- Create: `web/js/classifier.js`
- Modify: `web/js/main.js`

**Step 1: Create classifier.js**

```javascript
// Pose Classifier using TensorFlow.js
// Port of Python PoseClassifier

export const POSE_LABELS = ['SHOOT', 'SHIELD', 'RELOAD'];
const NUM_LANDMARKS = 33;
const NUM_FEATURES = NUM_LANDMARKS * 4; // x, y, z, visibility
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const CONFIDENCE_THRESHOLD = 0.7;
const MIN_VISIBILITY = 0.5;

export class PoseClassifier {
  constructor() {
    this.model = null;
    this.trainingSamples = []; // {features: Float32Array, labelIdx: number}
  }

  isReady() {
    return this.model !== null;
  }

  extractFeatures(landmarks) {
    // Get shoulder positions for normalization
    const lShoulder = landmarks[LEFT_SHOULDER];
    const rShoulder = landmarks[RIGHT_SHOULDER];

    // Need both shoulders visible
    if (lShoulder.visibility < MIN_VISIBILITY || rShoulder.visibility < MIN_VISIBILITY) {
      return null;
    }

    // Body center (midpoint of shoulders)
    const centerX = (lShoulder.x + rShoulder.x) / 2;
    const centerY = (lShoulder.y + rShoulder.y) / 2;
    const centerZ = (lShoulder.z + rShoulder.z) / 2;

    // Shoulder width for scale normalization
    const shoulderWidth = Math.sqrt(
      Math.pow(rShoulder.x - lShoulder.x, 2) +
      Math.pow(rShoulder.y - lShoulder.y, 2)
    );

    if (shoulderWidth < 0.01) return null;

    // Extract and normalize features
    const features = new Float32Array(NUM_FEATURES);
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const lm = landmarks[i];
      const baseIdx = i * 4;

      if (lm.visibility >= MIN_VISIBILITY) {
        features[baseIdx] = (lm.x - centerX) / shoulderWidth;
        features[baseIdx + 1] = (lm.y - centerY) / shoulderWidth;
        features[baseIdx + 2] = (lm.z - centerZ) / shoulderWidth;
        features[baseIdx + 3] = 1.0; // presence flag
      } else {
        features[baseIdx] = 0;
        features[baseIdx + 1] = 0;
        features[baseIdx + 2] = 0;
        features[baseIdx + 3] = 0;
      }
    }

    return features;
  }

  addSample(landmarks, poseLabel) {
    const features = this.extractFeatures(landmarks);
    if (!features) return false;

    const labelIdx = POSE_LABELS.indexOf(poseLabel);
    this.trainingSamples.push({ features, labelIdx });
    return true;
  }

  getSampleCount(poseLabel = null) {
    if (poseLabel === null) {
      return this.trainingSamples.length;
    }
    const labelIdx = POSE_LABELS.indexOf(poseLabel);
    return this.trainingSamples.filter(s => s.labelIdx === labelIdx).length;
  }

  clearTrainingData() {
    this.trainingSamples = [];
  }

  async train(epochs = 200) {
    if (this.trainingSamples.length < 10) {
      console.error('Need at least 10 samples to train');
      return false;
    }

    // Prepare data
    const xs = tf.tensor2d(
      this.trainingSamples.map(s => Array.from(s.features)),
      [this.trainingSamples.length, NUM_FEATURES]
    );
    const ys = tf.oneHot(
      tf.tensor1d(this.trainingSamples.map(s => s.labelIdx), 'int32'),
      POSE_LABELS.length
    );

    // Create model: matches Python architecture
    this.model = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [NUM_FEATURES], units: 128, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: POSE_LABELS.length, activation: 'softmax' }),
      ]
    });

    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    // Train
    await this.model.fit(xs, ys, {
      epochs,
      batchSize: 32,
      shuffle: true,
      verbose: 0,
    });

    xs.dispose();
    ys.dispose();

    console.log('Classifier trained');
    return true;
  }

  predict(landmarks) {
    if (!this.model) return 'NEUTRAL';

    const features = this.extractFeatures(landmarks);
    if (!features) return 'NEUTRAL';

    const input = tf.tensor2d([Array.from(features)], [1, NUM_FEATURES]);
    const prediction = this.model.predict(input);
    const probs = prediction.dataSync();
    const maxIdx = probs.indexOf(Math.max(...probs));
    const confidence = probs[maxIdx];

    input.dispose();
    prediction.dispose();

    if (confidence < CONFIDENCE_THRESHOLD) {
      return 'NEUTRAL';
    }

    return POSE_LABELS[maxIdx];
  }

  async saveToStorage() {
    if (!this.model) return false;
    await this.model.save('indexeddb://pose-classifier');
    console.log('Classifier saved to IndexedDB');
    return true;
  }

  async loadFromStorage() {
    try {
      this.model = await tf.loadLayersModel('indexeddb://pose-classifier');
      console.log('Classifier loaded from IndexedDB');
      return true;
    } catch (e) {
      console.log('No saved classifier found');
      return false;
    }
  }
}
```

**Step 2: Update main.js to use classifier**

Add import:
```javascript
import { PoseClassifier, POSE_LABELS } from './classifier.js';
```

Add to state:
```javascript
classifier: null,
```

Add initialization in init():
```javascript
state.classifier = new PoseClassifier();
state.classifierReady = await state.classifier.loadFromStorage();

if (state.classifierReady) {
  document.getElementById('message').textContent = 'Ready to play!';
} else {
  document.getElementById('message').textContent = 'Calibration needed';
}
```

Update detectLoop to show predictions:
```javascript
async function detectLoop() {
  const video = document.getElementById('webcam');
  const landmarks = await state.poseDetector.detect(video);

  if (landmarks) {
    drawLandmarks(landmarks);

    if (state.classifierReady) {
      const pose = state.classifier.predict(landmarks);
      document.getElementById('message').textContent = pose;
    }
  }

  requestAnimationFrame(detectLoop);
}
```

**Step 3: Verify classifier structure loads**

Open in browser, verify "Calibration needed" message appears.

**Step 4: Commit**

```bash
git add web/js/
git commit -m "feat(web): add TensorFlow.js pose classifier with IndexedDB persistence"
```

---

## Task 5: Calibration Flow

**Files:**
- Create: `web/js/calibration.js`
- Modify: `web/js/main.js`
- Modify: `web/index.html`

**Step 1: Create calibration.js**

```javascript
// Calibration UI and flow
import { POSE_LABELS } from './classifier.js';

export class Calibration {
  constructor(classifier, poseDetector, videoElement, canvasElement) {
    this.classifier = classifier;
    this.poseDetector = poseDetector;
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.isRunning = false;
    this.onComplete = null;
  }

  async run() {
    this.isRunning = true;
    this.classifier.clearTrainingData();

    const countdownDuration = 5000; // 5 seconds to get ready
    const recordDuration = 3000;    // 3 seconds of recording

    for (let poseIdx = 0; poseIdx < POSE_LABELS.length; poseIdx++) {
      const poseName = POSE_LABELS[poseIdx];

      // Countdown phase
      await this.countdown(poseName, poseIdx, countdownDuration);
      if (!this.isRunning) return false;

      // Recording phase
      await this.record(poseName, poseIdx, recordDuration);
      if (!this.isRunning) return false;
    }

    // Train the model
    this.showMessage('Training...', '#ffff00');
    const success = await this.classifier.train();

    if (success) {
      await this.classifier.saveToStorage();
      this.showMessage('Calibration Complete!', '#00ff00');
      await this.wait(2000);
    } else {
      this.showMessage('Training failed', '#ff0000');
      await this.wait(2000);
    }

    this.isRunning = false;
    return success;
  }

  async countdown(poseName, poseIdx, duration) {
    const startTime = Date.now();

    while (Date.now() - startTime < duration) {
      if (!this.isRunning) return;

      const remaining = Math.ceil((duration - (Date.now() - startTime)) / 1000);

      this.clearCanvas();
      this.drawText(`Get ready: ${poseName}`, -60, '#00ffff', 1.5);
      this.drawText(remaining.toString(), 20, '#ffffff', 3);
      this.drawText(`Pose ${poseIdx + 1}/${POSE_LABELS.length}`, 100, '#888888', 0.8);

      await this.wait(50);
    }
  }

  async record(poseName, poseIdx, duration) {
    const startTime = Date.now();
    let samples = 0;

    while (Date.now() - startTime < duration) {
      if (!this.isRunning) return;

      const remaining = ((duration - (Date.now() - startTime)) / 1000).toFixed(1);
      const landmarks = await this.poseDetector.detect(this.video);

      if (landmarks && this.classifier.addSample(landmarks, poseName)) {
        samples++;
      }

      this.clearCanvas();
      this.drawText(`Recording ${poseName}`, -40, '#00ff00', 1.2);
      this.drawText(`${remaining}s`, 20, '#00ff00', 2);
      this.drawText(`Samples: ${samples}`, 80, '#cccccc', 0.8);

      await this.wait(50);
    }
  }

  cancel() {
    this.isRunning = false;
  }

  clearCanvas() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  drawText(text, yOffset, color, scale) {
    const centerX = this.canvas.width / 2;
    const centerY = this.canvas.height / 2 + yOffset;
    const fontSize = Math.floor(40 * scale);

    this.ctx.font = `bold ${fontSize}px sans-serif`;
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';

    // Shadow
    this.ctx.fillStyle = '#000000';
    this.ctx.fillText(text, centerX + 2, centerY + 2);

    // Main text
    this.ctx.fillStyle = color;
    this.ctx.fillText(text, centerX, centerY);
  }

  showMessage(text, color) {
    this.clearCanvas();
    this.drawText(text, 0, color, 1.5);
  }

  wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

**Step 2: Update main.js for calibration**

Add import:
```javascript
import { Calibration } from './calibration.js';
```

Add calibration handler:
```javascript
function setupCalibration() {
  const btn = document.getElementById('calibrate-btn');
  btn.addEventListener('click', async () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('overlay');

    const calibration = new Calibration(
      state.classifier,
      state.poseDetector,
      video,
      canvas
    );

    btn.disabled = true;
    btn.textContent = 'Calibrating...';

    const success = await calibration.run();

    btn.disabled = false;
    btn.textContent = 'Calibrate';

    if (success) {
      state.classifierReady = true;
      document.getElementById('message').textContent = 'Ready to play!';
    }
  });
}
```

Call in init after pose is ready:
```javascript
setupCalibration();
```

**Step 3: Verify calibration works**

Open browser, click Calibrate, follow prompts, verify model saves.

**Step 4: Commit**

```bash
git add web/js/
git commit -m "feat(web): add calibration flow with countdown and recording"
```

---

## Task 6: RL Opponent (TensorFlow.js)

**Files:**
- Create: `web/js/opponent.js`

**Step 1: Create opponent.js**

```javascript
// RL Opponent using TensorFlow.js
// Port of Python RLOpponent

const ACTIONS = ['L', 'B', 'S']; // Load, Block, Shoot
const MAX_BULLETS = 10;
const HISTORY_LEN = 5;
const INPUT_SIZE = 32;
const HIDDEN_SIZE = 64;

// Payoff matrix: PAYOFF[aiAction][oppAction] = AI's reward
const PAYOFF = {
  'L': { 'L': 0.0, 'B': 0.0, 'S': -1.0 },
  'B': { 'L': 0.0, 'B': 0.0, 'S': 0.1 },
  'S': { 'L': 1.0, 'B': -0.1, 'S': 0.0 },
};

function toFeatures(myBullets, oppBullets, oppHistory, myHistory) {
  const features = new Float32Array(INPUT_SIZE);
  let idx = 0;

  features[idx++] = myBullets / MAX_BULLETS;
  features[idx++] = oppBullets / MAX_BULLETS;

  // One-hot encode last N opponent actions
  for (let i = 0; i < HISTORY_LEN; i++) {
    if (i < oppHistory.length) {
      const action = oppHistory[oppHistory.length - 1 - i];
      features[idx++] = action === 'L' ? 1 : 0;
      features[idx++] = action === 'B' ? 1 : 0;
      features[idx++] = action === 'S' ? 1 : 0;
    } else {
      idx += 3;
    }
  }

  // One-hot encode last N AI actions
  for (let i = 0; i < HISTORY_LEN; i++) {
    if (i < myHistory.length) {
      const action = myHistory[myHistory.length - 1 - i];
      features[idx++] = action === 'L' ? 1 : 0;
      features[idx++] = action === 'B' ? 1 : 0;
      features[idx++] = action === 'S' ? 1 : 0;
    } else {
      idx += 3;
    }
  }

  return features;
}

function createModel() {
  return tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [INPUT_SIZE], units: HIDDEN_SIZE, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.1 }),
      tf.layers.dense({ units: HIDDEN_SIZE, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.1 }),
      tf.layers.dense({ units: 3, activation: 'softmax' }),
    ]
  });
}

function getValidActions(bullets) {
  return bullets > 0 ? ['L', 'B', 'S'] : ['L', 'B'];
}

function selectAction(probs, aiBullets, oppBullets, explorationRate = 0.1) {
  let [pL, pB, pS] = probs;

  // Adjust if opponent can't shoot
  if (oppBullets <= 0) {
    pL = pL + pS * 0.7;
    pB = pB + pS * 0.3;
    pS = 0;
  }

  // Calculate expected value for each action
  const expectedValues = {};
  for (const action of ACTIONS) {
    expectedValues[action] =
      pL * PAYOFF[action]['L'] +
      pB * PAYOFF[action]['B'] +
      pS * PAYOFF[action]['S'];
  }

  const validActions = getValidActions(aiBullets);

  // Exploration
  if (Math.random() < explorationRate) {
    return validActions[Math.floor(Math.random() * validActions.length)];
  }

  // Pick best valid action
  return validActions.reduce((best, action) =>
    expectedValues[action] > expectedValues[best] ? action : best
  );
}

export class RLOpponent {
  constructor() {
    this.persistentModel = null;
    this.sessionModel = null;
    this.buffer = [];           // Persistent experience buffer
    this.sessionBuffer = [];    // Session-only buffer
    this.myHistory = [];
    this.oppHistory = [];
    this.roundsPlayed = 0;
  }

  async init() {
    await this.loadFromStorage();
    if (!this.persistentModel) {
      this.persistentModel = createModel();
      this.persistentModel.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'categoricalCrossentropy',
      });
    }
  }

  resetGame() {
    this.myHistory = [];
    this.oppHistory = [];
  }

  resetSession() {
    this.sessionBuffer = [];
    this.sessionModel = null;
    this.resetGame();
  }

  getAction(myBullets, oppBullets) {
    const features = toFeatures(myBullets, oppBullets, this.oppHistory, this.myHistory);
    const input = tf.tensor2d([Array.from(features)], [1, INPUT_SIZE]);

    // Get persistent model prediction
    const persistentPred = this.persistentModel.predict(input);
    let probs = persistentPred.dataSync();

    // Blend with session model if available
    if (this.sessionModel && this.sessionBuffer.length >= 10) {
      const sessionPred = this.sessionModel.predict(input);
      const sessionProbs = sessionPred.dataSync();
      const sessionWeight = Math.min(0.6, this.sessionBuffer.length / 50);

      probs = probs.map((p, i) => (1 - sessionWeight) * p + sessionWeight * sessionProbs[i]);
      sessionPred.dispose();
    }

    input.dispose();
    persistentPred.dispose();

    return selectAction(Array.from(probs), myBullets, oppBullets);
  }

  async update(myAction, oppAction) {
    const features = toFeatures(0, 0, this.oppHistory, this.myHistory);

    // Update histories
    this.myHistory.push(myAction);
    this.oppHistory.push(oppAction);
    this.roundsPlayed++;

    // Add to buffers
    const actionIdx = ACTIONS.indexOf(oppAction);
    this.buffer.push({ features: Array.from(features), actionIdx });
    this.sessionBuffer.push({ features: Array.from(features), actionIdx });

    // Keep buffer bounded
    if (this.buffer.length > 500) this.buffer.shift();

    // Train persistent model
    await this.trainStep(this.persistentModel, this.buffer);

    // Train/create session model
    if (this.sessionBuffer.length >= 10) {
      if (!this.sessionModel) {
        this.sessionModel = createModel();
        this.sessionModel.compile({
          optimizer: tf.train.adam(0.05),
          loss: 'categoricalCrossentropy',
        });
      }
      await this.trainStep(this.sessionModel, this.sessionBuffer, 10);
    }
  }

  async trainStep(model, buffer, epochs = 2) {
    if (buffer.length < 4) return;

    const batchSize = Math.min(16, buffer.length);
    const batch = buffer.length <= batchSize ? buffer :
      [...buffer].sort(() => Math.random() - 0.5).slice(0, batchSize);

    const xs = tf.tensor2d(batch.map(b => b.features), [batch.length, INPUT_SIZE]);
    const ys = tf.oneHot(tf.tensor1d(batch.map(b => b.actionIdx), 'int32'), 3);

    await model.fit(xs, ys, { epochs, verbose: 0 });

    xs.dispose();
    ys.dispose();
  }

  async saveToStorage() {
    if (this.persistentModel) {
      await this.persistentModel.save('indexeddb://rl-opponent');
      localStorage.setItem('rl-opponent-buffer', JSON.stringify(this.buffer));
      console.log('Opponent saved to IndexedDB');
    }
  }

  async loadFromStorage() {
    try {
      this.persistentModel = await tf.loadLayersModel('indexeddb://rl-opponent');
      this.persistentModel.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'categoricalCrossentropy',
      });

      const savedBuffer = localStorage.getItem('rl-opponent-buffer');
      if (savedBuffer) {
        this.buffer = JSON.parse(savedBuffer);
      }

      console.log('Opponent loaded from IndexedDB');
      return true;
    } catch (e) {
      console.log('No saved opponent found');
      return false;
    }
  }
}
```

**Step 2: Verify opponent compiles**

Import in main.js temporarily to check for errors:
```javascript
import { RLOpponent } from './opponent.js';
```

**Step 3: Commit**

```bash
git add web/js/opponent.js
git commit -m "feat(web): add RL opponent with persistent and session learning"
```

---

## Task 7: Game Logic

**Files:**
- Create: `web/js/game.js`

**Step 1: Create game.js**

```javascript
// Game state and logic
// Port of Python game logic

export const MAX_BULLETS = 10;
export const ACTION_NAMES = { 'L': 'RELOAD', 'B': 'SHIELD', 'S': 'SHOOT' };

export class GameState {
  constructor() {
    this.p1Bullets = 0;
    this.p2Bullets = 0;
    this.winner = null; // null=ongoing, 0=draw, 1=p1 wins, 2=p2 wins
  }

  get isTerminal() {
    return this.winner !== null;
  }

  clone() {
    const s = new GameState();
    s.p1Bullets = this.p1Bullets;
    s.p2Bullets = this.p2Bullets;
    s.winner = this.winner;
    return s;
  }
}

export function getValidActions(bullets) {
  return bullets > 0 ? ['L', 'B', 'S'] : ['L', 'B'];
}

export function step(state, p1Action, p2Action) {
  if (state.isTerminal) return state;

  const newState = state.clone();

  // Determine shots and blocks
  const p1Shot = p1Action === 'S' && state.p1Bullets > 0;
  const p2Shot = p2Action === 'S' && state.p2Bullets > 0;
  const p1Blocked = p1Action === 'B';
  const p2Blocked = p2Action === 'B';

  // Update bullet counts
  if (p1Action === 'L') {
    newState.p1Bullets = Math.min(state.p1Bullets + 1, MAX_BULLETS);
  } else if (p1Shot) {
    newState.p1Bullets = state.p1Bullets - 1;
  }

  if (p2Action === 'L') {
    newState.p2Bullets = Math.min(state.p2Bullets + 1, MAX_BULLETS);
  } else if (p2Shot) {
    newState.p2Bullets = state.p2Bullets - 1;
  }

  // Determine hits
  const p1Hit = p2Shot && !p1Blocked;
  const p2Hit = p1Shot && !p2Blocked;

  if (p1Hit && p2Hit) {
    newState.winner = 0; // Draw
  } else if (p1Hit) {
    newState.winner = 2; // P2 wins
  } else if (p2Hit) {
    newState.winner = 1; // P1 wins
  }

  return newState;
}

export function poseToAction(pose, bullets) {
  const mapping = {
    'SHOOT': 'S',
    'SHIELD': 'B',
    'RELOAD': 'L',
  };
  const action = mapping[pose];

  // Can't shoot without bullets
  if (action === 'S' && bullets <= 0) {
    return null;
  }
  return action;
}
```

**Step 2: Commit**

```bash
git add web/js/game.js
git commit -m "feat(web): add game state and logic"
```

---

## Task 8: Game Loop Integration

**Files:**
- Modify: `web/js/main.js`
- Modify: `web/index.html`

**Step 1: Add start button to HTML**

In index.html, add after calibrate button:
```html
<button id="start-btn" disabled>Start Game</button>
```

**Step 2: Rewrite main.js with full game loop**

```javascript
// 007 Pose Web App - Main Entry Point

import { PoseDetector } from './pose-detector.js';
import { PoseClassifier, POSE_LABELS } from './classifier.js';
import { Calibration } from './calibration.js';
import { RLOpponent } from './opponent.js';
import { GameState, step, poseToAction, getValidActions, ACTION_NAMES, MAX_BULLETS } from './game.js';

const state = {
  webcamReady: false,
  poseReady: false,
  classifierReady: false,
  poseDetector: null,
  classifier: null,
  opponent: null,
  gameState: null,
  roundNum: 0,
  wins: 0,
  losses: 0,
  draws: 0,
  isPlaying: false,
};

// DOM elements
let video, canvas, ctx, messageEl, roundEl, playerBulletsEl, aiBulletsEl, winsEl, lossesEl;

async function init() {
  console.log('Initializing 007 Pose...');

  // Cache DOM elements
  video = document.getElementById('webcam');
  canvas = document.getElementById('overlay');
  ctx = canvas.getContext('2d');
  messageEl = document.getElementById('message');
  roundEl = document.getElementById('round-num');
  playerBulletsEl = document.getElementById('player-bullets');
  aiBulletsEl = document.getElementById('ai-bullets');
  winsEl = document.getElementById('wins');
  lossesEl = document.getElementById('losses');

  // Initialize webcam
  state.webcamReady = await initWebcam();
  if (!state.webcamReady) return;

  // Initialize pose detector
  messageEl.textContent = 'Loading pose model...';
  state.poseDetector = new PoseDetector();
  await state.poseDetector.init();
  state.poseReady = true;

  // Initialize classifier
  state.classifier = new PoseClassifier();
  state.classifierReady = await state.classifier.loadFromStorage();

  // Initialize opponent
  state.opponent = new RLOpponent();
  await state.opponent.init();

  // Setup UI
  setupButtons();
  updateUI();

  if (state.classifierReady) {
    messageEl.textContent = 'Ready to play!';
    document.getElementById('start-btn').disabled = false;
  } else {
    messageEl.textContent = 'Calibration needed';
  }

  // Start detection loop
  detectLoop();
}

async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: 'user' }
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    console.log('Webcam ready');
    return true;
  } catch (err) {
    console.error('Webcam error:', err);
    messageEl.textContent = 'Camera access required';
    return false;
  }
}

function setupButtons() {
  document.getElementById('calibrate-btn').addEventListener('click', runCalibration);
  document.getElementById('start-btn').addEventListener('click', startGame);
}

async function runCalibration() {
  const btn = document.getElementById('calibrate-btn');
  const startBtn = document.getElementById('start-btn');

  btn.disabled = true;
  startBtn.disabled = true;
  btn.textContent = 'Calibrating...';

  const calibration = new Calibration(state.classifier, state.poseDetector, video, canvas);
  const success = await calibration.run();

  btn.disabled = false;
  btn.textContent = 'Calibrate';

  if (success) {
    state.classifierReady = true;
    startBtn.disabled = false;
    messageEl.textContent = 'Ready to play!';
  }
}

async function startGame() {
  if (!state.classifierReady) {
    messageEl.textContent = 'Calibrate first!';
    return;
  }

  state.isPlaying = true;
  state.gameState = new GameState();
  state.roundNum = 0;
  state.opponent.resetGame();

  document.getElementById('start-btn').disabled = true;
  document.getElementById('calibrate-btn').disabled = true;

  await gameLoop();
}

async function gameLoop() {
  while (!state.gameState.isTerminal) {
    state.roundNum++;
    updateUI();

    // Countdown
    await countdown(2000);

    // Get player pose
    const playerAction = await capturePlayerAction();
    if (playerAction === null) {
      // Game cancelled
      endGame(null);
      return;
    }

    // Get AI action
    const aiAction = state.opponent.getAction(state.gameState.p2Bullets, state.gameState.p1Bullets);

    // Execute turn
    const newState = step(state.gameState, playerAction, aiAction);

    // Update opponent
    await state.opponent.update(aiAction, playerAction);

    // Show result
    await showRoundResult(playerAction, aiAction);

    state.gameState = newState;
  }

  // Game over
  endGame(state.gameState.winner);
}

async function countdown(duration) {
  const startTime = Date.now();

  while (Date.now() - startTime < duration) {
    const remaining = Math.ceil((duration - (Date.now() - startTime)) / 1000);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (remaining <= 0) {
      drawCenteredText('POSE!', '#00ff00', 3);
    } else {
      drawCenteredText(remaining.toString(), '#00ffff', 4);
    }

    // Show valid moves
    const valid = getValidActions(state.gameState.p1Bullets);
    const validNames = valid.map(a => ACTION_NAMES[a]).join(', ');
    drawText(`Valid: ${validNames}`, 20, canvas.height - 30, '#cccccc', 0.8);

    await wait(50);
  }
}

async function capturePlayerAction() {
  let pose = 'NEUTRAL';
  let action = null;

  const startTime = Date.now();
  const timeout = 5000; // 5 second timeout

  while (action === null && Date.now() - startTime < timeout) {
    const landmarks = await state.poseDetector.detect(video);

    if (landmarks) {
      pose = state.classifier.predict(landmarks);
      action = poseToAction(pose, state.gameState.p1Bullets);
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawCenteredText('POSE!', '#00ff00', 3);

    if (pose !== 'NEUTRAL' && action === null) {
      drawCenteredText('No bullets!', '#ff0000', 1, 80);
    }

    await wait(50);
  }

  return action;
}

async function showRoundResult(playerAction, aiAction) {
  const duration = 2000;
  const startTime = Date.now();

  while (Date.now() - startTime < duration) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawCenteredText(`You: ${ACTION_NAMES[playerAction]}`, '#00ffff', 1.2, -50);
    drawCenteredText('vs', '#888888', 0.8, 0);
    drawCenteredText(`AI: ${ACTION_NAMES[aiAction]}`, '#ff6b6b', 1.2, 50);

    await wait(50);
  }
}

function endGame(winner) {
  state.isPlaying = false;

  let resultText, resultColor;

  if (winner === null) {
    resultText = 'Game Cancelled';
    resultColor = '#888888';
  } else if (winner === 0) {
    resultText = 'DRAW!';
    resultColor = '#ffff00';
    state.draws++;
  } else if (winner === 1) {
    resultText = 'YOU WIN!';
    resultColor = '#00ff00';
    state.wins++;
  } else {
    resultText = 'AI WINS!';
    resultColor = '#ff0000';
    state.losses++;
  }

  // Save opponent learning
  state.opponent.saveToStorage();

  // Show result for 3 seconds
  const showResult = async () => {
    const duration = 3000;
    const startTime = Date.now();

    while (Date.now() - startTime < duration) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawCenteredText(resultText, resultColor, 2.5);
      drawCenteredText(`Game ended in ${state.roundNum} rounds`, '#cccccc', 0.8, 80);
      await wait(50);
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    updateUI();
    messageEl.textContent = 'Ready to play!';
    document.getElementById('start-btn').disabled = false;
    document.getElementById('calibrate-btn').disabled = false;
  };

  showResult();
}

function updateUI() {
  roundEl.textContent = state.roundNum;
  winsEl.textContent = state.wins;
  lossesEl.textContent = state.losses;

  if (state.gameState) {
    playerBulletsEl.textContent = '|'.repeat(state.gameState.p1Bullets) +
                                   '.'.repeat(MAX_BULLETS - state.gameState.p1Bullets);
    aiBulletsEl.textContent = '|'.repeat(state.gameState.p2Bullets) +
                               '.'.repeat(MAX_BULLETS - state.gameState.p2Bullets);
  } else {
    playerBulletsEl.textContent = '.'.repeat(MAX_BULLETS);
    aiBulletsEl.textContent = '.'.repeat(MAX_BULLETS);
  }
}

async function detectLoop() {
  if (!state.isPlaying) {
    const landmarks = await state.poseDetector.detect(video);

    if (landmarks && state.classifierReady) {
      const pose = state.classifier.predict(landmarks);
      if (pose !== 'NEUTRAL') {
        messageEl.textContent = pose;
      }
    }
  }

  requestAnimationFrame(detectLoop);
}

// Drawing helpers
function drawCenteredText(text, color, scale, yOffset = 0) {
  const fontSize = Math.floor(50 * scale);
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2 + yOffset;

  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  ctx.fillStyle = '#000000';
  ctx.fillText(text, centerX + 3, centerY + 3);

  ctx.fillStyle = color;
  ctx.fillText(text, centerX, centerY);
}

function drawText(text, x, y, color, scale) {
  const fontSize = Math.floor(24 * scale);
  ctx.font = `${fontSize}px sans-serif`;
  ctx.textAlign = 'left';
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Start the app
init().catch(console.error);
```

**Step 3: Verify full game works**

1. Open in browser
2. Click Calibrate, record poses
3. Click Start Game
4. Play through a game

**Step 4: Commit**

```bash
git add web/
git commit -m "feat(web): integrate full game loop with UI"
```

---

## Task 9: Audio Feedback

**Files:**
- Create: `web/js/audio.js`
- Modify: `web/js/main.js`

**Step 1: Create audio.js**

```javascript
// Audio feedback using Web Audio API

let audioCtx = null;

function getAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioCtx;
}

export function playTone(frequency, duration = 0.15) {
  const ctx = getAudioContext();
  const oscillator = ctx.createOscillator();
  const gainNode = ctx.createGain();

  oscillator.connect(gainNode);
  gainNode.connect(ctx.destination);

  oscillator.frequency.value = frequency;
  oscillator.type = 'sine';

  gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration);

  oscillator.start(ctx.currentTime);
  oscillator.stop(ctx.currentTime + duration);
}

export function playCountdownTone() {
  playTone(440, 0.12);
}

export function playPoseTone() {
  playTone(880, 0.2);
}

export async function playWinSound() {
  const notes = [523, 659, 784, 1047];
  for (const freq of notes) {
    playTone(freq, 0.15);
    await new Promise(r => setTimeout(r, 150));
  }
}

export async function playLoseSound() {
  const notes = [392, 349, 311, 262];
  for (const freq of notes) {
    playTone(freq, 0.2);
    await new Promise(r => setTimeout(r, 200));
  }
}

export function playDrawSound() {
  playTone(440, 0.3);
  setTimeout(() => playTone(440, 0.3), 350);
}

export function playCompareSound() {
  playTone(660, 0.1);
}
```

**Step 2: Add audio to main.js**

Import:
```javascript
import { playCountdownTone, playPoseTone, playWinSound, playLoseSound, playDrawSound, playCompareSound } from './audio.js';
```

Add sounds to countdown:
```javascript
// In countdown function, when number changes:
if (remaining !== lastRemaining) {
  if (remaining <= 0) {
    playPoseTone();
  } else {
    playCountdownTone();
  }
  lastRemaining = remaining;
}
```

Add sound to showRoundResult:
```javascript
playCompareSound();
```

Add sounds to endGame:
```javascript
if (winner === 0) {
  playDrawSound();
} else if (winner === 1) {
  await playWinSound();
} else if (winner === 2) {
  await playLoseSound();
}
```

**Step 3: Commit**

```bash
git add web/js/
git commit -m "feat(web): add audio feedback with Web Audio API"
```

---

## Task 10: Final Polish and Deployment

**Files:**
- Modify: `web/index.html`
- Modify: `web/css/style.css`

**Step 1: Add favicon and meta tags**

In index.html head:
```html
<meta name="description" content="007 Pose - A pose-based game where you battle an AI using hand gestures">
<meta name="theme-color" content="#1a1a2e">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔫</text></svg>">
```

**Step 2: Add loading state handling**

```css
/* Add to style.css */
.loading {
  opacity: 0.5;
  pointer-events: none;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

**Step 3: Test locally**

```bash
cd web && python3 -m http.server 8000
```

Open http://localhost:8000 and test full flow.

**Step 4: Commit all changes**

```bash
git add web/
git commit -m "feat(web): add final polish and meta tags"
```

**Step 5: Push to GitHub**

```bash
git push origin main
```

**Step 6: Enable GitHub Pages**

1. Go to repo Settings → Pages
2. Source: "Deploy from a branch"
3. Branch: main, folder: /web
4. Save

**Step 7: Verify deployment**

Visit `https://<username>.github.io/007-pose/` after a few minutes.

**Step 8: Final commit with deployment info**

```bash
git add .
git commit -m "docs: update with GitHub Pages deployment"
```

---

## Summary

Total tasks: 10
Estimated implementation: Multiple focused sessions

Key milestones:
1. Tasks 1-3: Project setup and pose detection working
2. Tasks 4-5: Classifier with calibration
3. Tasks 6-7: RL opponent and game logic
4. Task 8: Full game playable
5. Tasks 9-10: Polish and deployment
