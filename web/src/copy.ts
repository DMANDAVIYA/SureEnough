export const HOOK = `Brain decoders always guess. SureEnough knows when it has seen enough, says "not sure" when it isn't, and keeps its promise about how often it's wrong.`;
export const LABEL =
  'Replayed public lab EEG (THINGS-EEG2). Not live. Each repeat is a 100 ms flash in a rapid sequence.';
export const CREDIT =
  'EEG: THINGS-EEG2 (Gifford et al. 2022), CC-BY 4.0 · Object photos: THINGSplus, CC0.';
export const BADGE =
  "Promise: for this person, the true object is outside SureEnough's final answer at most 2 in 10 times (checked with 90% confidence).";
export const REVEAL_CAPTION = "A different photo of the same thing. The real stimulus can't be shown (licence).";
export const NOT_COVERED = 'not covered by the promise';

const repeats = (n: number): string => `${n} ${n === 1 ? 'repeat' : 'repeats'}`;
export const stoppedAfter = (n: number): string => `Stopped after ${repeats(n)}`;
export const inTen = (rate: number): string => `${(rate * 10).toFixed(1)} in 10`;
