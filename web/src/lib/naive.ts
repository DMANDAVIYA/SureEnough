import type { Table } from '../data/schema';
import { mean, noiseTolerantCeil } from './stats';

export interface NaiveOutcome {
  evalRisk: number;
}

function conformalThresholds(calibration: number[][][], alpha: number): number[][] {
  const rank = noiseTolerantCeil((calibration.length + 1) * (1 - alpha));
  return calibration[0].map((steps, r) =>
    steps.map((_, t) =>
      rank > calibration.length ? Infinity : Float64Array.from(calibration, (image) => image[r][t]).sort()[rank - 1],
    ),
  );
}

function streamMiss(aTrue: number[], top3: number[], thresholds: number[]): number {
  const crossing = top3.findIndex((mass, t) => mass > thresholds[t]);
  const t = crossing === -1 ? top3.length - 1 : crossing;
  return Number(aTrue[t] > thresholds[t]);
}

export function naivePolicy(table: Table, calIdx: number[], evalIdx: number[], alpha: number): NaiveOutcome {
  const { a_true_u16: aTrue, top3_u16: top3 } = table.naive;
  const thresholds = conformalThresholds(calIdx.map((i) => aTrue[i]), alpha);
  const imageRisk = (i: number): number => mean(aTrue[i].map((stream, r) => streamMiss(stream, top3[i][r], thresholds[r])));
  return { evalRisk: mean(evalIdx.map(imageRisk)) };
}
