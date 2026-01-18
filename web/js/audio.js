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
