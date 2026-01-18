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
