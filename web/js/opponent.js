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
