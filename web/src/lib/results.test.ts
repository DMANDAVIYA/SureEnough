import { describe, expect, it } from 'vitest';
import type { OutcomeMix, Results, SubjectSummary } from '../data/schema';
import { findingsOf, meanOutcomeMix, raceOf, subjectSummary } from './results';

const mix = (single: number, category: number, list: number, full: number): OutcomeMix => ({ single, category, list, full });

const summaryOf = (overrides: Partial<SubjectSummary>): SubjectSummary => ({
  eval_risk: 0.12,
  conservativeness: 0.09,
  mean_stop: 6.5,
  mean_set_size: 3.4,
  fixed_points: [
    [40.2, 1],
    [21.5, 2],
    [9.1, 4],
    [3.1, 8],
    [1.9, 16],
    [1.7, 20],
  ],
  fixed_eval_risk: { '1': 0.15, '2': 0.14, '4': 0.13, '8': 0.12, '16': 0.11, '20': 0.1 },
  h1_ratio: 0.8,
  h2a: { cp_upper: 0.043 },
  outcome_mix: mix(0.3, 0.1, 0.5, 0.1),
  ...overrides,
});

const results: Results = {
  per_subject: {
    '1': summaryOf({ h1_ratio: 0.6, conservativeness: 0.05, h2a: { cp_upper: 0.051 } }),
    '2': summaryOf({ h1_ratio: 0.9, conservativeness: 0.07, outcome_mix: mix(0.5, 0.1, 0.3, 0.1) }),
    '3': summaryOf({ h1_ratio: 0.75, conservativeness: 0.09, outcome_mix: mix(0.1, 0.4, 0.4, 0.1) }),
  },
  pvalues: { H1: 0.042, H2b: 0.001 },
  holm: { H1: false, H2b: true },
  h2b_count: 3,
};

describe('subjectSummary', () => {
  it('looks a person up by number in the per-subject map', () => {
    expect(subjectSummary(results, 2)).toBe(results.per_subject['2']);
    expect(subjectSummary(results, 9)).toBeUndefined();
  });
});

describe('raceOf', () => {
  it('races SureEnough against the fewest-repeat fixed rule that answers at least as narrowly', () => {
    expect(raceOf(summaryOf({}))).toEqual({
      sure: { repeats: 6.5, wrong: 0.12, size: 3.4 },
      fixed: { repeats: 8, wrong: 0.12, size: 3.1 },
    });
  });

  it('falls back to the longest fixed rule when none answers as narrowly', () => {
    expect(raceOf(summaryOf({ mean_set_size: 1.2 })).fixed).toEqual({ repeats: 20, wrong: 0.1, size: 1.7 });
  });

  it('does not depend on the order of the fixed points', () => {
    const summary = summaryOf({});
    expect(raceOf({ ...summary, fixed_points: [...summary.fixed_points].reverse() })).toEqual(raceOf(summary));
  });
});

describe('findingsOf', () => {
  it('summarises the confirmatory tests from the exported p-values, Holm decisions and per-person checks', () => {
    const findings = findingsOf(results);
    expect(findings.h1).toEqual({ ratio: 0.75, p: 0.042, holds: false, parity: true });
    expect(findings.h2a.cpUpper).toBe(0.051);
    expect(findings.h2a.conservativeness).toBeCloseTo(0.07, 12);
    expect(findings.h2b).toEqual({ count: 3, p: 0.001 });
  });

  it('calls H1 efficient only when Holm keeps it, and drops parity when SureEnough needs clearly more repeats', () => {
    const slower = { ...results, per_subject: { '1': summaryOf({ h1_ratio: 1.4 }) } };
    expect(findingsOf(slower).h1).toMatchObject({ holds: false, parity: false });
    const kept = { ...results, holm: { H1: true, H2b: true } };
    expect(findingsOf(kept).h1).toMatchObject({ holds: true, parity: false });
  });

  it('passes the PE vs CLIP-H line through only when the export has it', () => {
    expect(findingsOf(results).peVsClip).toBeNull();
    expect(findingsOf({ ...results, pe_vs_clip: { pe: 0.31, clip: 0.25 } }).peVsClip).toEqual({ pe: 0.31, clip: 0.25 });
  });
});

describe('meanOutcomeMix', () => {
  it('averages each outcome share over people', () => {
    const average = meanOutcomeMix(results);
    expect(average.single).toBeCloseTo(0.3, 12);
    expect(average.category).toBeCloseTo(0.2, 12);
    expect(average.list).toBeCloseTo(0.4, 12);
    expect(average.full).toBeCloseTo(0.1, 12);
  });
});
