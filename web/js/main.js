// 007 Pose Web App - Main Entry Point

console.log('007 Pose loading...');

// App state
const state = {
  webcamReady: false,
  poseReady: false,
  classifierReady: false,
  gameState: null,
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

async function init() {
  console.log('Initializing app...');

  state.webcamReady = await initWebcam();
  if (!state.webcamReady) return;

  document.getElementById('message').textContent = 'Camera ready!';

  // TODO: Initialize MediaPipe Pose
  // TODO: Initialize classifier
  // TODO: Start game loop
}

init().catch(console.error);
