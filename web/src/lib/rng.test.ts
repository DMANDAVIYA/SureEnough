import { describe, expect, it } from 'vitest';
import { mulberry32, permutation } from './rng';

const draws = (random: () => number, count: number): number[] => Array.from({ length: count }, random);

describe('mulberry32', () => {
  it('repeats the same sequence for the same seed', () => {
    expect(draws(mulberry32(42), 5)).toEqual(draws(mulberry32(42), 5));
  });

  it('gives different sequences for different seeds', () => {
    expect(draws(mulberry32(1), 5)).not.toEqual(draws(mulberry32(2), 5));
  });

  it('stays in [0, 1)', () => {
    const values = draws(mulberry32(7), 10000);
    expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...values)).toBeLessThan(1);
  });
});

describe('permutation', () => {
  it('contains every index exactly once', () => {
    expect([...permutation(50, mulberry32(3))].sort((a, b) => a - b)).toEqual(
      Array.from({ length: 50 }, (_, i) => i),
    );
  });

  it('is reproducible from the seed', () => {
    expect(permutation(20, mulberry32(9))).toEqual(permutation(20, mulberry32(9)));
  });

  it('draws all orders of three items about equally often', () => {
    const random = mulberry32(11);
    const counts = new Map<string, number>();
    for (let draw = 0; draw < 6000; draw++) {
      const key = permutation(3, random).join('');
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    expect(counts.size).toBe(6);
    for (const count of counts.values()) expect(Math.abs(count - 1000)).toBeLessThan(150);
  });
});
