import { describe, expect, it } from 'vitest';
import { stackedOpacities } from './blend';

const visibleShares = (opacities: number[]): number[] =>
  opacities.map((opacity, k) => opacities.slice(k + 1).reduce((share, above) => share * (1 - above), opacity));

describe('stackedOpacities', () => {
  it('shows a lone candidate fully', () => {
    expect(stackedOpacities([0.7])).toEqual([1]);
  });

  it('leaves each stacked layer visible in proportion to its weight', () => {
    const weights = [0.5, 0.25, 0.15, 0.1];
    visibleShares(stackedOpacities(weights)).forEach((share, k) => expect(share).toBeCloseTo(weights[k], 12));
  });

  it('normalises weights that do not sum to one', () => {
    visibleShares(stackedOpacities([3, 1])).forEach((share, k) => expect(share).toBeCloseTo([0.75, 0.25][k], 12));
  });
});
