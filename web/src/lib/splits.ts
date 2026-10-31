import type { Manifest, Table } from '../data/schema';
import { mapInBackground } from './background';
import { recalibrate } from './ltt';
import { naivePolicy } from './naive';
import { permutation } from './rng';
import { mean } from './stats';

export interface Split {
  cal: number[];
  eval: number[];
}

export interface Risks {
  sure: number;
  naive: number;
}

const ascending = (values: number[]): number[] => [...values].sort((a, b) => a - b);

export function randomSplit(n: number, random: () => number): Split {
  const order = permutation(n, random);
  const half = Math.floor(n / 2);
  return { cal: ascending(order.slice(0, half)), eval: ascending(order.slice(half)) };
}

export function meanRisks(table: Table, splits: Split[], alpha: number, delta: number): Risks {
  return {
    sure: mean(splits.map((s) => recalibrate(table, s.cal, s.eval, alpha, delta).evalRisk)),
    naive: mean(splits.map((s) => naivePolicy(table, s.cal, s.eval, alpha).evalRisk)),
  };
}

const averageRisks = (risks: Risks[]): Risks => ({
  sure: mean(risks.map((r) => r.sure)),
  naive: mean(risks.map((r) => r.naive)),
});

export function meanRisksInBackground(
  tables: Table[],
  splits: Split[],
  alpha: number,
  delta: number,
  done: (perTable: Risks[]) => void,
): () => void {
  const jobs = tables.flatMap((table) => splits.map((split) => ({ table, split })));
  return mapInBackground(
    jobs,
    ({ table, split }) => meanRisks(table, [split], alpha, delta),
    (perSplit) => done(tables.map((_, k) => averageRisks(perSplit.slice(k * splits.length, (k + 1) * splits.length)))),
  );
}

const pickFrom = <T>(values: T[], random: () => number): T => values[Math.floor(random() * values.length)];

export function pickMystery(manifest: Manifest, random: () => number): { subject: number; i: number } {
  const subject = pickFrom(manifest.subjects, random);
  return { subject, i: pickFrom(manifest.pinned[String(subject)].eval, random) };
}
