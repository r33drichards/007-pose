# 007 Pose Web App Design

## Goal

Deploy 007 Pose as a shareable web app that coworkers can play from a browser link. Single-player, client-side only, hosted on GitHub Pages.

## Architecture

Fully client-side web app with no server dependencies.

```
┌─────────────────────────────────────────────────────┐
│                    Browser                          │
├─────────────────────────────────────────────────────┤
│  Webcam (getUserMedia)                              │
│       ↓                                             │
│  MediaPipe Pose (JS) → landmarks                    │
│       ↓                                             │
│  Pose Classifier (TensorFlow.js)                    │
│       ↓                                             │
│  Game Logic + RL Opponent (JS port)                 │
│       ↓                                             │
│  Canvas UI (render game state)                      │
│       ↓                                             │
│  IndexedDB (persist calibration + opponent model)   │
└─────────────────────────────────────────────────────┘
```

## File Structure

```
007-pose/
├── web/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── main.js           # Entry point, orchestration
│   │   ├── pose-detector.js  # MediaPipe wrapper
│   │   ├── classifier.js     # Pose classification (TF.js)
│   │   ├── opponent.js       # RL opponent (TF.js)
│   │   ├── game.js           # Game loop and state
│   │   ├── ui.js             # Canvas rendering
│   │   └── storage.js        # IndexedDB persistence
│   └── assets/
│       └── (sounds if any)
└── (existing Python files unchanged)
```

## Pose Detection & Classification

**MediaPipe Pose (JavaScript)**
- Official JS/WASM build from Google
- Same 33 body landmarks as Python version
- 30+ FPS in modern browsers

**Classifier (TensorFlow.js)**

Port of current PyTorch MLP:
```
Input (66 features: 33 landmarks × 2 coords)
  → Dense(128, relu)
  → Dense(3, softmax)
  → Output (SHOOT/SHIELD/RELOAD)
```

Using TensorFlow.js instead of ONNX export because:
- Better browser support and documentation
- Need training APIs for in-browser calibration
- Model is simple enough to rewrite

**Calibration flow:**
1. User enters calibration mode
2. Records ~20 samples of each pose
3. Model trains in-browser (few seconds)
4. Saved to IndexedDB for next session

## RL Opponent

Port of `RLOpponent` class with two learning layers:

```javascript
class RLOpponent {
  persistentModel  // TF.js model, saved to IndexedDB
  sessionModel     // TF.js model, in-memory only

  predict(moveHistory) → counterMove
  learn(actualMove)    → update weights
}
```

**Counter logic:**
- Predict player's next move from history
- Pick move that beats it

**Persistence:**
- Load persistent model from IndexedDB on page load
- Update both models after each round
- Save persistent model on page close

## UI Layout

```
┌─────────────────────────────────────────┐
│  007 POSE                    [Calibrate]│
├─────────────────────────────────────────┤
│                                         │
│         ┌───────────────────┐           │
│         │                   │           │
│         │   Webcam Feed     │           │
│         │   + Pose Overlay  │           │
│         │                   │           │
│         └───────────────────┘           │
│                                         │
│   YOU: ████░░ (ammo)    THEM: ████░░    │
│   Detected: SHIELD      Last: SHOOT     │
│                                         │
│         [ READY ] or [ 3... 2... 1... ] │
│                                         │
│   Score: You 3 - 2 Opponent             │
└─────────────────────────────────────────┘
```

**Tech choices:**
- Single `<canvas>` for webcam + overlays
- Vanilla JS (no framework needed)
- Minimal CSS styling

## Game Flow

1. First visit → prompt calibration (or skip if saved data exists)
2. Calibration → record 20 samples per pose, train, save
3. Game → countdown, detect pose, resolve round, update scores
4. Continuous play until user closes tab

## Deployment

**GitHub Pages:**
1. Push `web/` folder to repo
2. Settings → Pages → Source: main branch, `/web` folder
3. Live at `https://<username>.github.io/007-pose/`

**Dependencies (CDN):**
- `@mediapipe/pose`
- `@tensorflow/tfjs`

No build step required.
