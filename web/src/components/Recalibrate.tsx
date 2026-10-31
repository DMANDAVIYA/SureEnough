import { useState } from 'react';
import { inTen } from '../copy';
import type { Table } from '../data/schema';
import { mulberry32 } from '../lib/rng';
import { meanRisks, randomSplit, type Risks } from '../lib/splits';
import PromiseChart, { ChartLegend } from './PromiseChart';

interface Props {
  table: Table | null;
  alpha: number;
  delta: number;
  images: number;
}

interface Run extends Risks {
  seed: number;
  ms: number;
}

export default function Recalibrate({ table, alpha, delta, images }: Props) {
  const [run, setRun] = useState<Run | null>(null);
  const next = () => {
    if (!table) return;
    const seed = (run?.seed ?? 0) + 1;
    const started = performance.now();
    const risks = meanRisks(table, [randomSplit(images, mulberry32(seed))], alpha, delta);
    setRun({ seed, ...risks, ms: performance.now() - started });
  };
  return (
    <section className="panel recalibrate" aria-label="Live recalibration">
      <h2>Check the promise yourself</h2>
      <p className="panel-note">
        Calibrate on a random half of this person's {images} images, then count the misses on the other half.
      </p>
      <button type="button" className="button" onClick={next} disabled={!table}>
        New random split
      </button>
      {run && (
        <div className="recalibrate-result" aria-live="polite">
          <p>
            Split {run.seed}: delivered <strong data-testid="delivered">{inTen(run.sure)}</strong> against a promise of
            at most {Math.round(alpha * 10)} in 10. The naive repeated look delivered <strong>{inTen(run.naive)}</strong>.
          </p>
          <ChartLegend />
          <PromiseChart rows={[{ label: `Split ${run.seed}`, sure: run.sure, naive: run.naive }]} alpha={alpha} title="Delivered error on this split" />
          <p className="panel-note">Recalibrated in your browser in {Math.max(1, Math.round(run.ms))} ms.</p>
        </div>
      )}
    </section>
  );
}
