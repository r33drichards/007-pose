// 007 Pose Web App - Main Entry Point

import { PoseDetector } from './pose-detector.js';
import { PoseClassifier, POSE_LABELS } from './classifier.js';
import { Calibration } from './calibration.js';

console.log('007 Pose loading...');

// App state
const state = {
  webcamReady: false,
  poseReady: false,
  classifierReady: false,
  gameState: null,
  poseDetector: null,
  classifier: null,
};

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

async function initPose() {
  state.poseDetector = new PoseDetector();
  await state.poseDetector.init();
  console.log('Pose detector ready');
  return true;
}

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

async function init() {
  console.log('Initializing app...');

  state.webcamReady = await initWebcam();
  if (!state.webcamReady) return;

  document.getElementById('message').textContent = 'Loading pose model...';
  state.poseReady = await initPose();

  // Initialize classifier
  state.classifier = new PoseClassifier();
  state.classifierReady = await state.classifier.loadFromStorage();

  if (state.classifierReady) {
    document.getElementById('message').textContent = 'Ready to play!';
  } else {
    document.getElementById('message').textContent = 'Calibration needed';
  }

  setupCalibration();

  // Start detection loop
  detectLoop();
}

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

init().catch(console.error);
