import type { Example } from '../data/schema';

export const MS_PER_REPEAT = 600;
const BLEND_LIMIT = 6;

export interface BlendLayer {
  i: number;
  weight: number;
}

export interface Frame {
  repeat: number;
  setSize: number;
  setMembers: number[] | null;
  blend: BlendLayer[];
  stopped: boolean;
}

export interface Player {
  stopAt: number;
  step(repeat: number): Frame;
}

export interface Clock {
  now(): number;
  frame(callback: () => void): () => void;
}

const browserClock: Clock = {
  now: () => performance.now(),
  frame: (callback) => {
    const id = requestAnimationFrame(callback);
    return () => cancelAnimationFrame(id);
  },
};

function blendOf(ranked: Example['top'][number]): BlendLayer[] {
  const total = ranked.reduce((sum, [, p]) => sum + p, 0);
  return ranked.map(([i, p]) => ({ i, weight: p / total }));
}

export function createPlayer(example: Example): Player {
  const stopAt = example.stopped_at;
  return {
    stopAt,
    step(repeat) {
      const shown = Math.min(Math.max(repeat, 1), stopAt);
      const setSize = example.set_size[shown - 1];
      const ranked = example.top[shown - 1];
      return {
        repeat: shown,
        setSize,
        setMembers: setSize <= ranked.length ? ranked.slice(0, setSize).map(([i]) => i) : null,
        blend: blendOf(ranked.slice(0, Math.min(setSize, BLEND_LIMIT))),
        stopped: repeat >= stopAt,
      };
    },
  };
}

export function play(
  stopAt: number,
  onRepeat: (repeat: number) => void,
  clock: Clock = browserClock,
  msPerRepeat = MS_PER_REPEAT,
): () => void {
  const start = clock.now();
  let shown = 0;
  let cancelFrame = () => {};
  const tick = () => {
    const repeat = Math.min(stopAt, 1 + Math.floor((clock.now() - start) / msPerRepeat));
    if (repeat !== shown) {
      shown = repeat;
      onRepeat(repeat);
    }
    if (repeat < stopAt) cancelFrame = clock.frame(tick);
  };
  tick();
  return () => cancelFrame();
}
