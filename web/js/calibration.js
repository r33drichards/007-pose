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
