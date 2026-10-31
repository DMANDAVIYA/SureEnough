import { describe, expect, it } from 'vitest';
import type { Example } from '../data/schema';
import { type Clock, createPlayer, play } from './player';

type Ranked = [number, number][];

const exampleOf = (top: Ranked[], setSize: number[], stoppedAt: number): Example => ({
  top,
  set_size: setSize,
  stopped_at: stoppedAt,
  eeg: top.map(() => []),
  claim: { kind: 'list', text: "It's one of these 2.", category: null },
  narration: null,
  provenance: null,
  pe_guess: 'wolf',
  impression: null,
  truth_in_set: true,
});

const truncated = exampleOf(
  [
    [[4, 9000], [1, 8000], [2, 7000], [0, 6000]],
    [[4, 20000], [1, 15000], [2, 9000], [0, 4000]],
    [[4, 40000], [1, 9000], [0, 5000], [7, 4000]],
    [[4, 50000], [1, 9000], [0, 3000], [7, 2000]],
  ],
  [150, 4, 2, 1],
  3,
);

describe('createPlayer', () => {
  it('stops exactly at the exported stop', () => {
    expect(createPlayer(truncated).stopAt).toBe(3);
    expect(createPlayer(exampleOf(truncated.top, [150, 4, 2, 1], 4)).stopAt).toBe(4);
  });

  it('reports the exported set size at every repeat, even beyond the exported ranking', () => {
    const player = createPlayer(truncated);
    expect([1, 2, 3].map((repeat) => player.step(repeat).setSize)).toEqual([150, 4, 2]);
  });

  it('lists the members only when the exported ranking covers the whole set', () => {
    const player = createPlayer(truncated);
    expect(player.step(1).setMembers).toBeNull();
    expect(player.step(2).setMembers).toEqual([4, 1, 2, 0]);
    expect(player.step(3).setMembers).toEqual([4, 1]);
  });

  it('freezes on the stopped frame after the stop', () => {
    const player = createPlayer(truncated);
    expect(player.step(2).stopped).toBe(false);
    expect(player.step(3).stopped).toBe(true);
    expect(player.step(20)).toEqual(player.step(3));
  });

  it('blends the leading members, weighted by their share of the shown likelihood', () => {
    const player = createPlayer(truncated);
    expect(player.step(3).blend).toEqual([
      { i: 4, weight: 40000 / 49000 },
      { i: 1, weight: 9000 / 49000 },
    ]);
  });

  it('caps the blend at six leading candidates when the set is larger', () => {
    const wide = exampleOf([Array.from({ length: 30 }, (_, k): [number, number] => [k, 3000 - k])], [120], 1);
    const blend = createPlayer(wide).step(1).blend;
    expect(blend.map((layer) => layer.i)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(blend.reduce((sum, layer) => sum + layer.weight, 0)).toBeCloseTo(1, 12);
  });
});

function fakeClock() {
  let time = 0;
  let pending: (() => void) | null = null;
  const clock: Clock = {
    now: () => time,
    frame: (callback) => {
      pending = callback;
      return () => {
        pending = null;
      };
    },
  };
  const advance = (ms: number) => {
    time += ms;
    const callback = pending;
    pending = null;
    callback?.();
  };
  return { clock, advance, waiting: () => pending !== null };
}

describe('play', () => {
  it('shows the first repeat at once, then one repeat per interval, and stops at the stop', () => {
    const { clock, advance, waiting } = fakeClock();
    const seen: number[] = [];
    play(3, (repeat) => seen.push(repeat), clock, 600);
    advance(300);
    advance(300);
    advance(700);
    expect(seen).toEqual([1, 2, 3]);
    expect(waiting()).toBe(false);
  });

  it('jumps straight to the right repeat after a long frame', () => {
    const { clock, advance } = fakeClock();
    const seen: number[] = [];
    play(20, (repeat) => seen.push(repeat), clock, 600);
    advance(6100);
    expect(seen).toEqual([1, 11]);
  });

  it('stops calling back once cancelled', () => {
    const { clock, advance, waiting } = fakeClock();
    const seen: number[] = [];
    const cancel = play(5, (repeat) => seen.push(repeat), clock, 600);
    cancel();
    advance(600);
    expect(seen).toEqual([1]);
    expect(waiting()).toBe(false);
  });
});
