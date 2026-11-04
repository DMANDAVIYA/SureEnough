import type { ClaimKind, OutcomeMix, Results, SubjectSummary } from '../data/schema';
import { mean } from './stats';

const PARITY_RATIO = 1.1;
const KINDS: ClaimKind[] = ['single', 'category', 'list', 'full'];

export interface RaceLane {
  repeats: number;
  wrong: number;
  size: number;
}

export interface RaceLanes {
  sure: RaceLane;
  fixed: RaceLane;
}

export interface Findings {
  h1: { ratio: number; p: number; holds: boolean; parity: boolean };
  h2a: { cpUpper: number; conservativeness: number };
  h2b: { count: number; p: number; holds: boolean; naiveRisk: number | null };
  bestGuessRisk: number | null;
  h5a: { n: number; p: number; holds: boolean; hedged: number; top1: number; truth: number } | null;
  peVsClip: { pe: number; clip: number } | null;
}

const median = (values: number[]): number => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

const meanOf = (people: SubjectSummary[], key: 'naive_pop_risk' | 'uncalibrated_risk'): number | null =>
  people.every((s) => s[key] !== undefined) ? mean(people.map((s) => s[key]!)) : null;

export const subjectSummary = (results: Results, subject: number): SubjectSummary | undefined =>
  results.per_subject[String(subject)];

export function raceOf(summary: SubjectSummary): RaceLanes {
  const sure = { repeats: summary.mean_stop, wrong: summary.eval_risk, size: summary.mean_set_size };
  const lanes = summary.fixed_points
    .map(([size, k]) => ({ repeats: k, wrong: summary.fixed_eval_risk[String(k)], size }))
    .sort((a, b) => a.repeats - b.repeats);
  return { sure, fixed: lanes.find((lane) => lane.size <= sure.size) ?? lanes[lanes.length - 1] };
}

export function findingsOf(results: Results): Findings {
  const people = Object.values(results.per_subject);
  const ratio = median(people.map((s) => s.h1_ratio));
  const holds = results.holm.H1;
  return {
    h1: { ratio, p: results.pvalues.H1, holds, parity: !holds && ratio <= PARITY_RATIO },
    h2a: {
      cpUpper: Math.max(...people.map((s) => s.h2a.cp_upper)),
      conservativeness: mean(people.map((s) => s.conservativeness)),
    },
    h2b: { count: results.h2b_count, p: results.pvalues.H2b, holds: results.holm.H2b, naiveRisk: meanOf(people, 'naive_pop_risk') },
    bestGuessRisk: meanOf(people, 'uncalibrated_risk'),
    h5a: results.h5a
      ? {
          n: results.h5a.n,
          p: results.h5a.p,
          holds: results.holm.H5a ?? false,
          hedged: results.h5a.h5b_means.composite_hedged,
          top1: results.h5a.h5b_means.composite_top1,
          truth: results.h5a.h5b_means.composite_answer_key,
        }
      : null,
    peVsClip: results.pe_vs_clip ?? null,
  };
}

export const meanOutcomeMix = (results: Results): OutcomeMix => {
  const people = Object.values(results.per_subject);
  return Object.fromEntries(KINDS.map((kind) => [kind, mean(people.map((s) => s.outcome_mix[kind]))])) as OutcomeMix;
};
