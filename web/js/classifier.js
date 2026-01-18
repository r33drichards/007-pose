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
