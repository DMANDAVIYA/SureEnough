import type { Table } from '../data/schema';
import { mean, noiseTolerantCeil } from './stats';

export interface Recalibration {
  lamIdx: number;
  evalRisk: number;
}

const xlogy = (x: number, y: number): number => (x === 0 ? 0 : x * Math.log(y));

const bernoulliKl = (a: number, b: number): number => xlogy(a, a / b) + xlogy(1 - a, (1 - a) / (1 - b));

const logSumExp = (logs: number[]): number => {
  const peak = Math.max(...logs);
  return peak + Math.log(logs.reduce((sum, log) => sum + Math.exp(log - peak), 0));
};

function binomialCdf(k: number, n: number, p: number): number {
  if (k < 0) return 0;
  if (k >= n) return 1;
  const logOdds = Math.log(p) - Math.log1p(-p);
  const logPmf = [n * Math.log1p(-p)];
  for (let i = 0; i < k; i++) logPmf.push(logPmf[i] + Math.log((n - i) / (i + 1)) + logOdds);
  return Math.exp(logSumExp(logPmf));
}

export function hbPvalue(r: number, n: number, alpha: number): number {
  const hoeffding = Math.exp(-n * bernoulliKl(Math.min(r, alpha), alpha));
  const bentkus = Math.E * binomialCdf(noiseTolerantCeil(n * r), n, alpha);
  return Math.min(hoeffding, bentkus, 1);
}

export function fixedSequence(pvalues: Iterable<number>, delta: number): number {
  let rejected = 0;
  for (const p of pvalues) {
    if (p > delta) break;
    rejected++;
  }
  return Math.max(rejected - 1, 0);
}

const meanAt = (rows: number[][], scale: number, idx: number[], column: number): number =>
  mean(idx.map((i) => rows[i][column] / scale));

function* columnPvalues(lossCounts: number[][], R: number, calIdx: number[], alpha: number): Generator<number> {
  for (let j = 0; j < lossCounts[0].length; j++) yield hbPvalue(meanAt(lossCounts, R, calIdx, j), calIdx.length, alpha);
}

export function calibrate(lossCounts: number[][], R: number, calIdx: number[], alpha: number, delta: number): number {
  if (new Set(calIdx).size !== calIdx.length) throw new Error('Calibration images must be distinct');
  return fixedSequence(columnPvalues(lossCounts, R, calIdx, alpha), delta);
}

export function recalibrate(
  table: Table,
  calIdx: number[],
  evalIdx: number[],
  alpha: number,
  delta: number,
): Recalibration {
  const R = table.naive.a_true_u16[0].length;
  const lamIdx = calibrate(table.loss_count, R, calIdx, alpha, delta);
  return { lamIdx, evalRisk: meanAt(table.loss_count, R, evalIdx, lamIdx) };
}
