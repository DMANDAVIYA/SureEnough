import { useEffect, useMemo, useState } from 'react';
import Pips from '../components/Pips';
import PromiseChart, { ChartLegend, type RiskRow } from '../components/PromiseChart';
import { inTen } from '../copy';
import { useManifest } from '../data/manifest';
import { ready, useResource } from '../data/useResource';
import { loadTable } from '../lib/data';
import { findingsOf } from '../lib/results';
import { mulberry32 } from '../lib/rng';
import { meanRisksInBackground, randomSplit, type Risks } from '../lib/splits';

const SPLITS_PER_PERSON = 20;
const STRICTNESS = [0.1, 0.2, 0.3];

const percent = (x: number): string => `${Math.round(x * 100)}%`;
const pValue = (p: number): string => (p < 0.001 ? '< 0.001' : `= ${p.toFixed(3)}`);
const outOfTen = (alpha: number): number => Math.round(alpha * 10);

function RiskTable({ rows }: { rows: RiskRow[] }) {
  return (
    <details className="table-view">
      <summary>Show as a table</summary>
      <table>
        <thead>
          <tr>
            <th scope="col">Person</th>
            <th scope="col">SureEnough</th>
            <th scope="col">Naive repeated look</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <th scope="row">{row.label}</th>
              <td>{inTen(row.sure)}</td>
              <td>{inTen(row.naive)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

export default function Results() {
  const manifest = useManifest();
  const loaded = useResource('tables', () => Promise.all(manifest.subjects.map(loadTable)));
  const tables = ready(loaded);
  const [alpha, setAlpha] = useState(manifest.alpha);
  const splits = useMemo(
    () => Array.from({ length: SPLITS_PER_PERSON }, (_, k) => randomSplit(manifest.concepts.length, mulberry32(k + 1))),
    [manifest],
  );
  const [risksByAlpha, setRisksByAlpha] = useState<Record<number, Risks[]>>({});
  const risks = risksByAlpha[alpha];
  useEffect(() => {
    if (!tables || risks) return;
    return meanRisksInBackground(tables, splits, alpha, manifest.delta, (computed) =>
      setRisksByAlpha((cache) => ({ ...cache, [alpha]: computed })),
    );
  }, [tables, splits, alpha, risks, manifest.delta]);
  const rows = risks?.map((risk, k) => ({ label: `Person ${manifest.subjects[k]}`, ...risk }));
  const { h1, h2a, h2b, peVsClip: pe } = useMemo(() => findingsOf(manifest.results), [manifest]);

  return (
    <div className="results">
      <section className="panel" aria-label="Promised vs delivered">
        <h2>Promised vs delivered error</h2>
        <p className="panel-note">
          Each person's error on held-out images, averaged over {SPLITS_PER_PERSON} random splits and recalibrated live
          in your browser.
        </p>
        <fieldset className="strictness">
          <legend>How strict is the promise?</legend>
          {STRICTNESS.map((option) => (
            <label key={option}>
              <input type="radio" name="strictness" checked={alpha === option} onChange={() => setAlpha(option)} />
              <Pips total={10} open={outOfTen(option)} />
              at most {outOfTen(option)} in 10
            </label>
          ))}
        </fieldset>
        <p className="panel-note">2 in 10 is the pre-registered promise. 1 and 3 in 10 are exploratory.</p>
        {loaded instanceof Error && (
          <p className="status" role="alert">
            The loss tables did not load ({loaded.message}).
          </p>
        )}
        {rows ? (
          <>
            <ChartLegend />
            <PromiseChart rows={rows} alpha={alpha} title="Promised vs delivered error per person" />
            <RiskTable rows={rows} />
          </>
        ) : (
          !(loaded instanceof Error) && <p className="status">Recalibrating…</p>
        )}
      </section>
      <section className="panel findings" aria-label="Findings">
        <h2>What the replays showed</h2>
        <p className="context">
          K = {manifest.K} repeats, within-subject, α = {manifest.alpha}, δ = {manifest.delta}, repeats from 4 sessions.
        </p>
        <ul>
          <li>Delivered error sat {inTen(h2a.conservativeness)} under the promise on average.</li>
          <li>
            Guarantee check: over 1,000 redrawn calibration sets per person, the promise failed in at most{' '}
            {percent(h2a.cpUpper)} of them for any person (95% upper bound). It is allowed to fail in {percent(manifest.delta)}.
          </li>
          <li>
            The naive repeated look broke the promise for {h2b.count} of {manifest.subjects.length} people (sign test p{' '}
            {pValue(h2b.p)}).
          </li>
          <li>
            For the median person, SureEnough used {h1.ratio.toFixed(2)}× the repeats of the best fixed rule at the same
            answer size
            (pre-registered target 0.7× or less, Wilcoxon p {pValue(h1.p)}).{h1.holds ? '' : ' The target was not met.'}
          </li>
          {pe && (
            <li>
              With the same linear decoder on held-out validation images, decoding into Perception Encoder space put the
              right object first {percent(pe.pe)} of the time after 4 repeats, against {percent(pe.clip)} for CLIP-H
              targets.
            </li>
          )}
        </ul>
      </section>
    </div>
  );
}
