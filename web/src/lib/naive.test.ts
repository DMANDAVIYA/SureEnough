import { describe, expect, it } from 'vitest';
import type { Table } from '../data/schema';
import { naivePolicy } from './naive';

const naiveTable = (aTrue: number[][][], top3: number[][][]): Table => ({
  loss_count: [],
  stop_x10: [],
  size_x10: [],
  naive: { a_true_u16: aTrue, top3_u16: top3 },
});

const handBuilt = naiveTable(
  [
    [[30000, 20000, 3000]],
    [[40000, 10000, 0]],
    [[50000, 30000, 5000]],
    [[45000, 25000, 4000]],
  ],
  [
    [[20000, 20000, 3000]],
    [[0, 0, 0]],
    [[0, 0, 0]],
    [[30000, 50000, 60000]],
  ],
);

describe('naivePolicy', () => {
  it('stops at the first step whose top-3 mass passes the threshold and scores a miss', () => {
    expect(naivePolicy(handBuilt, [0, 1, 2], [3], 0.5)).toEqual({ evalRisk: 1 });
  });

  it('runs to the budget and scores a hit when the threshold is never passed', () => {
    expect(naivePolicy(handBuilt, [1, 2, 3], [0], 0.5)).toEqual({ evalRisk: 0 });
  });

  it('never stops early when the conformal rank exceeds the calibration size', () => {
    expect(naivePolicy(handBuilt, [0, 1, 2], [3], 0.2)).toEqual({ evalRisk: 0 });
  });

  it('treats values equal to the threshold as neither a stop nor a miss and averages over streams', () => {
    const tied = naiveTable(
      [
        [[10, 20], [1, 1]],
        [[10, 20], [2, 2]],
        [[10, 20], [3, 3]],
        [[10, 20], [3, 0]],
      ],
      [
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[10, 21], [3, 0]],
      ],
    );
    expect(naivePolicy(tied, [0, 1, 2], [3], 0.5)).toEqual({ evalRisk: 0.5 });
  });
});
