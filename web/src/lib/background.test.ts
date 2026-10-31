import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mapInBackground } from './background';

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('mapInBackground', () => {
  it('works through one item per task, so the page can respond between items', () => {
    const seen: number[] = [];
    mapInBackground([1, 2, 3], (x) => seen.push(x), () => {});
    expect(seen).toEqual([]);
    vi.advanceTimersToNextTimer();
    expect(seen).toEqual([1]);
    vi.advanceTimersToNextTimer();
    expect(seen).toEqual([1, 2]);
  });

  it('reports every result in order once the last item is done', () => {
    const done = vi.fn();
    mapInBackground([1, 2, 3], (x) => x * 10, done);
    vi.runAllTimers();
    expect(done).toHaveBeenCalledOnce();
    expect(done).toHaveBeenCalledWith([10, 20, 30]);
  });

  it('does no more work and never reports once cancelled', () => {
    const work = vi.fn((x: number) => x);
    const done = vi.fn();
    const cancel = mapInBackground([1, 2, 3], work, done);
    vi.advanceTimersToNextTimer();
    cancel();
    vi.runAllTimers();
    expect(work).toHaveBeenCalledOnce();
    expect(done).not.toHaveBeenCalled();
  });
});
