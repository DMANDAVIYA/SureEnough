import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Manifest, Table } from '../data/schema';
import { recalibrate } from './ltt';
import { naivePolicy } from './naive';
import { mulberry32 } from './rng';
import { meanRisks, meanRisksInBackground, pickMystery, randomSplit } from './splits';

const range = (n: number): number[] => Array.from({ length: n }, (_, i) => i);

describe('randomSplit', () => {
  it('cuts the gallery into two disjoint sorted halves that cover every image', () => {
    const { cal, eval: evalIdx } = randomSplit(200, mulberry32(1));
    expect(cal).toHaveLength(100);
    expect(evalIdx).toHaveLength(100);
    expect([...cal, ...evalIdx].sort((a, b) => a - b)).toEqual(range(200));
    expect(cal).toEqual([...cal].sort((a, b) => a - b));
    expect(evalIdx).toEqual([...evalIdx].sort((a, b) => a - b));
  });

  it('is reproducible from the seed and differs across seeds', () => {
    expect(randomSplit(200, mulberry32(4))).toEqual(randomSplit(200, mulberry32(4)));
    expect(randomSplit(200, mulberry32(4))).not.toEqual(randomSplit(200, mulberry32(5)));
  });
});

const images = 24;
const table: Table = {
  loss_count: range(images).map((i) => [0, i % 4 === 0 ? 1 : 0, i % 2]),
  stop_x10: range(images).map(() => [200, 150, 80]),
  size_x10: range(images).map(() => [2000, 40, 15]),
  naive: {
    a_true_u16: range(images).map((i) => [[(i * 2654) % 65535, (i * 911) % 65535]]),
    top3_u16: range(images).map((i) => [[(i * 4099) % 65535, (i * 7919) % 65535]]),
  },
};

describe('meanRisks', () => {
  it('averages the delivered error of SureEnough and the naive method over the splits', () => {
    const splits = [1, 2, 3].map((seed) => randomSplit(images, mulberry32(seed)));
    const sure = splits.map((s) => recalibrate(table, s.cal, s.eval, 0.3, 0.1).evalRisk);
    const naive = splits.map((s) => naivePolicy(table, s.cal, s.eval, 0.3).evalRisk);
    const average = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
    expect(meanRisks(table, splits, 0.3, 0.1)).toEqual({ sure: average(sure), naive: average(naive) });
  });
});

describe('meanRisksInBackground', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  const shifted: Table = { ...table, loss_count: table.loss_count.map((row) => row.map((count) => 1 - count)) };
  const splits = [1, 2, 3].map((seed) => randomSplit(images, mulberry32(seed)));

  it('reports the same per-person averages as meanRisks', () => {
    vi.useFakeTimers();
    const done = vi.fn();
    meanRisksInBackground([table, shifted], splits, 0.3, 0.1, done);
    vi.runAllTimers();
    expect(done).toHaveBeenCalledWith([meanRisks(table, splits, 0.3, 0.1), meanRisks(shifted, splits, 0.3, 0.1)]);
  });

  it('recalibrates one split per task, so the page stays responsive', () => {
    vi.useFakeTimers();
    const done = vi.fn();
    meanRisksInBackground([table, shifted], splits, 0.3, 0.1, done);
    expect(vi.getTimerCount()).toBe(1);
    for (let task = 0; task < 2 * splits.length; task++) vi.advanceTimersToNextTimer();
    expect(done).not.toHaveBeenCalled();
    vi.advanceTimersToNextTimer();
    expect(done).toHaveBeenCalledOnce();
  });

  it('never reports once cancelled', () => {
    vi.useFakeTimers();
    const done = vi.fn();
    const cancel = meanRisksInBackground([table], splits, 0.3, 0.1, done);
    vi.advanceTimersToNextTimer();
    cancel();
    vi.runAllTimers();
    expect(done).not.toHaveBeenCalled();
  });
});

const manifest = {
  subjects: [3, 7],
  pinned: {
    '3': { cal: [0, 1], eval: [2, 5], lam_idx: 0 },
    '7': { cal: [2, 5], eval: [0, 1], lam_idx: 0 },
  },
} as unknown as Manifest;

describe('pickMystery', () => {
  it('only picks evaluation-half images of a listed subject', () => {
    const random = mulberry32(8);
    for (let draw = 0; draw < 50; draw++) {
      const { subject, i } = pickMystery(manifest, random);
      expect(manifest.pinned[String(subject)].eval).toContain(i);
    }
  });

  it('reaches every subject', () => {
    const random = mulberry32(9);
    const subjects = new Set(range(50).map(() => pickMystery(manifest, random).subject));
    expect([...subjects].sort()).toEqual([3, 7]);
  });
});
