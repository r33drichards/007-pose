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
