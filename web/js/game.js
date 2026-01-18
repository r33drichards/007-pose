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
