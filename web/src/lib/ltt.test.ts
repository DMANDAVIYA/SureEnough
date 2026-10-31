import { describe, expect, it } from 'vitest';
import type { LttFixture, Table } from '../data/schema';
import { calibrate, fixedSequence, hbPvalue, recalibrate } from './ltt';
import { naivePolicy } from './naive';

const COMMITTED_FIXTURE = './fixtures/ltt_fixture.json';
const fixtures = import.meta.glob<LttFixture>(
  ['./fixtures/ltt_fixture.json', '../../public/data/fixtures/ltt_fixture.json'],
  { eager: true, import: 'default' },
);

const calibrationCounts = Array.from({ length: 20 }, (_, i) => [0, 0, i === 0 ? 1 : 0, i < 4 ? 1 : 0, 0]);
const evaluationCounts = [
  [0, 2, 2, 2, 0],
  [0, 2, 1, 2, 0],
  [0, 2, 0, 1, 0],
  [0, 0, 0, 0, 0],
];
const lossCounts = [...calibrationCounts, ...evaluationCounts];
const calIdx = calibrationCounts.map((_, i) => i);
const evalIdx = evaluationCounts.map((_, i) => calibrationCounts.length + i);
const streams = 2;

const withEvaluationColumn = (values: number[]): number[][] => [
  ...calibrationCounts.map(() => [200, 200, 200, 200, 200]),
  ...values.map((value) => [200, 200, value, 200, 200]),
];

const table: Table = {
  loss_count: lossCounts,
  stop_x10: withEvaluationColumn([70, 50, 100, 200]),
  size_x10: withEvaluationColumn([10, 30, 15, 25]),
  naive: {
    a_true_u16: lossCounts.map(() => Array.from({ length: streams }, () => [0])),
    top3_u16: lossCounts.map(() => Array.from({ length: streams }, () => [0])),
  },
};

describe('hbPvalue', () => {
  it('equals the Hoeffding bound at zero observed risk', () => {
    expect(hbPvalue(0, 100, 0.2) / 0.8 ** 100).toBeCloseTo(1, 9);
  });

  it('is clipped to 1 at the target risk', () => {
    expect(hbPvalue(0.2, 100, 0.2)).toBe(1);
  });

  it('never decreases as the observed risk grows', () => {
    const pvalues = Array.from({ length: 101 }, (_, k) => hbPvalue(k / 200, 100, 0.2));
    pvalues.slice(1).forEach((p, k) => expect(p).toBeGreaterThanOrEqual(pvalues[k]));
  });

  it('matches the Python reference on intermediate risks', () => {
    expect(hbPvalue(0.025, 20, 0.2)).toBeCloseTo(0.05973237, 7);
    expect(hbPvalue(0.1, 20, 0.2)).toBeCloseTo(0.48008108, 7);
  });
});

describe('fixedSequence', () => {
  it('deploys the last rejection before the first non-rejection', () => {
    expect(fixedSequence([0.01, 0.02, 0.5, 0.01], 0.1)).toBe(1);
  });

  it('falls back to index 0 when nothing is rejected', () => {
    expect(fixedSequence([0.5, 0.01], 0.1)).toBe(0);
  });

  it('deploys the last index when everything is rejected', () => {
    expect(fixedSequence([0.01, 0.02, 0.03], 0.1)).toBe(2);
  });

  it('stops drawing p-values at the first non-rejection', () => {
    const drawn: number[] = [];
    function* pvalues() {
      for (const p of [0.01, 0.5, 0.01, 0.01]) {
        drawn.push(p);
        yield p;
      }
    }
    expect(fixedSequence(pvalues(), 0.1)).toBe(0);
    expect(drawn).toEqual([0.01, 0.5]);
  });
});

describe('calibrate', () => {
  it('picks the boldest certified index from the calibration rows only', () => {
    expect(calibrate(lossCounts, streams, calIdx, 0.2, 0.1)).toBe(2);
  });

  it('certifies less when the evaluation rows leak in', () => {
    expect(calibrate(lossCounts, streams, [...calIdx, ...evalIdx], 0.2, 0.1)).toBe(0);
  });

  it('rejects duplicate calibration images', () => {
    expect(() => calibrate(lossCounts, streams, [0, 0, ...calIdx.slice(2)], 0.2, 0.1)).toThrow();
  });
});

describe('recalibrate', () => {
  it('reports evaluation outcomes at the deployed index', () => {
    expect(recalibrate(table, calIdx, evalIdx, 0.2, 0.1)).toEqual({ lamIdx: 2, evalRisk: 0.375 });
  });
});

describe('python fixture parity', () => {
  it('has the committed Python-generated fixture', () => {
    expect(Object.keys(fixtures)).toContain(COMMITTED_FIXTURE);
  });

  it.each(Object.entries(fixtures))('reproduces the Python deployed index and eval risks in %s', (_, fixture) => {
    const { table: fixtureTable, alpha, delta, splits } = fixture;
    expect(new Set(splits.map((split) => split.lam_idx)).size).toBeGreaterThan(1);
    for (const split of splits) {
      const live = recalibrate(fixtureTable, split.cal, split.eval, alpha, delta);
      expect(live.lamIdx).toBe(split.lam_idx);
      expect(Math.abs(live.evalRisk - split.eval_risk)).toBeLessThanOrEqual(1e-12);
      const naive = naivePolicy(fixtureTable, split.cal, split.eval, alpha);
      expect(Math.abs(naive.evalRisk - split.naive_eval_risk)).toBeLessThanOrEqual(1e-12);
    }
  });
});
